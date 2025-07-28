#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, Linear, Sequential
from torch.nn import functional as F

from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.attention import Attention
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.dinov2_eomt_semantic_segmentation.scale_block import (
    ScaleBlock,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class DINOv2EoMTSemanticSegmentation(TaskModel):
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        num_queries: int,
        num_joint_blocks: int,
        backbone_weights: PathLike | None = None,
        backbone_freeze: bool = False,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(locals())
        # Disable drop path by default.
        args = {
            "drop_path_rate": 0.0,
        }
        if backbone_args is not None:
            args.update(backbone_args)

        # Get the backbone.
        self.backbone: DinoVisionTransformer = DINOV2_VIT_PACKAGE.get_model(
            model_name=model_name,
            model_args=args,
        )
        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        # TODO(Guarin, 07/25): Improve how mask tokens are handled for fine-tuning.
        # Should we drop them from the model? We disable grads here for DDP to work
        # without find_unused_parameters=True.
        self.backbone.mask_token.requires_grad = False

        # Load the backbone weights if a path is provided.
        # TODO(Thomas,07/2026): this should be done in the package.
        if backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

        if backbone_freeze:
            self.freeze_backbone()

        if len(self.backbone.blocks) < num_joint_blocks:
            raise ValueError(
                f"num_joint_blocks ({num_joint_blocks}) cannot be larger than the "
                f"number of blocks in the backbone ({len(self.backbone.blocks)})."
            )

        ### EoMT Specific parameters.
        self.num_queries = num_queries
        # Number of blocks that process queries and image tokens jointly.
        self.num_joint_blocks = num_joint_blocks
        self.queries = Embedding(num_queries, embed_dim)
        self.class_head = Linear(embed_dim, num_classes + 1)
        self.mask_head = Sequential(
            Linear(embed_dim, embed_dim),
            GELU(),
            Linear(embed_dim, embed_dim),
            GELU(),
            Linear(embed_dim, embed_dim),
        )

        num_upscale = max(1, int(math.log2(self.patch_size)) - 2)
        self.upscale = Sequential(
            *[ScaleBlock(embed_dim) for _ in range(num_upscale)],
        )

        # TODO(Guarin, 07/25): Move all attention mask handling to the train module.
        # Attention mask prob can be passed as argument to forward_train. No need to
        # store it as a parameter here.
        self.masked_attn_enabled = True
        self.register_buffer(
            "attn_mask_probs", torch.ones(self.num_joint_blocks), persistent=False
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # TODO(Guarin, 07/25): Update to return (masks, logits) tuple.
        mask_logits_per_layer, class_logits_per_layer = self.forward_train(x)
        return mask_logits_per_layer[0], class_logits_per_layer[0]

    # TODO(Guarin, 07/25): Refactor to take attn_mask_probs as input.
    def forward_train(self, x: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        B, C, H, W = x.shape
        patch_size = self.backbone.patch_size
        grid_size = (H // patch_size, W // patch_size)

        x = self.backbone.prepare_tokens_with_masks(x)  # type: ignore[no-untyped-call]
        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.backbone.blocks):
            if i == len(self.backbone.blocks) - self.num_joint_blocks:
                # Prepend query tokens.
                x = torch.cat(
                    (self.queries.weight[None, :, :].expand(x.shape[0], -1, -1), x),
                    dim=1,
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.backbone.blocks) - self.num_joint_blocks
            ):
                mask_logits, class_logits = self._predict(
                    self.backbone.norm(x), grid_size=grid_size
                )
                # TODO(Guarin, 07/25): Do we want to norm before appending? This is what
                # DINOv2 does.
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                # NOTE(Guarin, 07/25): Attention masking is enabled for training and
                # validation. We keep it enabled for validation to match metrics of the
                # original EoMT implementation. This results in significantly higher
                # validation mIoU during training. However, it would also make sense
                # to disable during validation as inference doesn't use attention
                # masking.
                # TODO(Guarin, 07/25): Disable for inference.
                attn_mask = torch.ones(
                    x.shape[0],
                    x.shape[1],
                    x.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                interpolated = F.interpolate(
                    input=mask_logits,
                    size=grid_size,
                    mode="bilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :,
                    : self.num_queries,
                    self.num_queries + 1 + self.backbone.num_register_tokens :,
                ] = interpolated > 0
                attn_mask = self._disable_attn_mask(
                    attn_mask=attn_mask,
                    prob=self.attn_mask_probs[
                        i - len(self.backbone.blocks) + self.num_joint_blocks
                    ],
                )

            # This mirrors forward of DINOv2 Block.
            if self.training and block.sample_drop_ratio > 0:
                x = x + block.drop_path1(
                    block.ls1(self._attn(block.attn, block.norm1(x), attn_mask))
                )
                x = x + block.drop_path1(block.ls2(block.mlp(block.norm2(x))))
            else:
                x = x + block.ls1(self._attn(block.attn, block.norm1(x), attn_mask))
                x = x + block.ls2(block.mlp(block.norm2(x)))

        mask_logits, class_logits = self._predict(
            self.backbone.norm(x), grid_size=grid_size
        )
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )

    def _predict(self, x: Tensor, grid_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        q = x[:, : self.num_queries, :]

        class_logits = self.class_head(q)

        # num queries + 1 class token + num register tokens
        x = x[:, self.num_queries + 1 + self.backbone.num_register_tokens :, :]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *grid_size)

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    # TODO(Guarin, 07/25): No need for attention mask handling in this module. Move it
    # to DINOv2SemanticSegmentationTrain.
    @torch.compiler.disable  # type: ignore[misc]
    def _disable_attn_mask(self, attn_mask: Tensor, prob: Tensor) -> Tensor:
        # prob is a scalar tensor.
        if prob < 1:
            random_queries = (
                torch.rand(
                    attn_mask.shape[0], self.num_queries, device=attn_mask.device
                )
                > prob
            )
            attn_mask[
                :,
                : self.num_queries,
                self.num_queries + 1 + self.backbone.num_register_tokens :,
            ][random_queries] = True

        return attn_mask

    # TODO(Guarin, 07/25): Add support for attention masks directly to Attention class?
    def _attn(self, module: Attention, x: Tensor, mask: Tensor | None) -> Tensor:
        # This mirrors DINOv2 Attention forward but with mask support.
        B, N, C = x.shape

        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * module.scale, qkv[1], qkv[2]

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = module.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = module.proj(x)
        x = module.proj_drop(x)
        return x

    def load_backbone_weights(self, path: PathLike) -> None:
        """
        Load backbone weights from a checkpoint file.

        Args:
            path: path to a .pt file, e.g., exported_last.pt.
        """
        # Check if the file exists.
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}")
            return

        # Load the checkpoint.
        state_dict = torch.load(path, map_location="cpu")

        # Load the state dict into the backbone.
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)

        # Log missing and unexpected keys.
        if missing or unexpected:
            if missing:
                logger.warning(f"Missing keys when loading backbone: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading backbone: {unexpected}")
        else:
            logger.info("Backbone weights loaded successfully.")

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        param_names = {name for name, _ in self.named_parameters()}
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("model."):
                name = name[len("model.") :]
                if name in param_names:
                    new_state_dict[name] = param
        self.load_state_dict(new_state_dict, strict=True)

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
