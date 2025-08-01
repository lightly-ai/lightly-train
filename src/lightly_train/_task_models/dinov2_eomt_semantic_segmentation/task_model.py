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
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import GELU, Embedding, Linear, Sequential
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
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
        classes: dict[int, str],
        class_ignore_index: int | None,
        num_queries: int,
        num_joint_blocks: int,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            model_name:
                The model name. For example "vits14-pretrain-eomt".
            classes:
                A dict mapping the class ID to the class name. The dict must only
                contain the classes that the model should predict. It must NOT contain
                classes that are in the dataset but should be ignored by the model.
            class_ignore_index:
                The class ID assigned to pixels that do not belong to any of the
                classes in `classes`. If None, the model will not ignore any classes and
                always assign a class to each pixel.
            num_queries:
                The number of query tokens to use in the model. This is the number of
                individual segments that the model will predict.
            num_joint_blocks:
                The number of blocks that process the query tokens and image tokens
                jointly.
            backbone_weights:
                The path to the DINOv2 backbone weights. The weights must be exported
                using LightlyTrain.
            backbone_args:
                Additional arguments to pass to the DINOv2 backbone.
        """
        super().__init__(locals(), ignore_args={"backbone_weights"})
        self.classes = classes
        self.class_ignore_index = class_ignore_index

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        # An additional class is added to represent "unknown/ignored classes" if needed.
        internal_class_to_class = list(self.classes.keys())
        if self.class_ignore_index is not None:
            internal_class_to_class.append(self.class_ignore_index)

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

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
        self.class_head = Linear(embed_dim, len(self.classes) + 1)
        self.mask_head = Sequential(
            Linear(embed_dim, embed_dim),
            GELU(),
            Linear(embed_dim, embed_dim),
            GELU(),
            Linear(embed_dim, embed_dim),
        )

        num_upscale = max(1, math.ceil(math.log2(self.patch_size)) - 2)
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

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> Tensor:
        """Returns the predicted mask for the given image.

        Args:
            image:
                The input image as a path, PIL image, or tensor. Tensors must have shape
                (C, H, W).

        Returns:
            The predicted mask as a tensor of shape (H, W). The values represent the
            class IDs as defined in the `classes` argument of your dataset. These
            classes are also stored in the `classes` attribute of the model.
            If your dataset contains ignored classes defined by the `ignore_classes`
            argument, the model will assign a special value for any pixels that it
            cannot assign to any of the known classes. This value is stored as
            `class_ignore_index` attribute of the model and is by default -100.
            If the dataset doesn't contain any ignored classes, the model will always
            assign a known class to each pixel.
        """
        if self.training:
            self.eval()

        # Load image
        x = file_helpers.as_image_tensor(image)
        image_h, image_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)
        # TODO(Guarin, 07/25): Save mean and std in the model.
        x = transforms_functional.normalize(
            x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # Resize shorter edge to 518
        # TODO(Guarin, 07/25): Make this configurable. Save default image size in the
        # model.
        x = transforms_functional.resize(x, size=[518])  # (C, H, W) -> (C, H', W')
        x = x.unsqueeze(0)  # (1, C, H', W')

        logits = self._forward_logits(x)  # (1, K+1, H', W'), K = len(self.classes)
        if self.class_ignore_index is None:
            # Restrict logits to known classes only.
            logits = logits[:, :-1]  # (1, K, H', W')
        logits = F.interpolate(
            logits, size=(image_h, image_w), mode="bilinear"
        )  # (1, K|K+1, H, W)

        masks = logits.argmax(dim=1)  # (1, H, W)
        # Map internal class IDs to class IDs.
        masks = self.internal_class_to_class[masks]  # (1, H, W)
        return masks[0]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Function used for ONNX export
        logits = self._forward_logits(x)  # (B, C, H, W)
        if self.class_ignore_index is None:
            # Restrict logits to known classes only.
            logits = logits[:, :-1]
        masks = logits.argmax(dim=1)  # (B, H, W)
        # Map internal class IDs to class IDs.
        masks = self.internal_class_to_class[masks]
        return masks, logits

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

    def tile(
        self, images: list[Tensor] | Tensor
    ) -> tuple[list[Tensor], list[tuple[int, int, int, bool]]]:
        crops, origins = [], []

        for i, image in enumerate(images):
            h, w = image.shape[-2:]
            long_side_size = max(h, w)
            short_side_size = min(h, w)

            # Is the image tall or wide?
            is_tall = h > w

            # By construction the short side size is equal to the crop size.
            crop_size = short_side_size
            num_crops = math.ceil(long_side_size / crop_size)
            overlap = num_crops * crop_size - long_side_size
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (crop_size - overlap_per_crop))
                end = start + crop_size

                # Image is tall.
                if is_tall:
                    crop = image[:, start:end, :]

                # Image is wide.
                else:
                    crop = image[:, :, start:end]

                # Store the crop.
                crops.append(crop)

                # Store the position of the crop.
                origins.append((i, start, end, is_tall))

        return crops, origins

    def untile(
        self,
        crop_logits: Tensor,
        origins: list[tuple[int, int, int, bool]],
        image_sizes: list[tuple[int, int]],
    ) -> list[Tensor]:
        logit_sums, logit_counts = [], []

        # Initialize the tensors containing the final predictions.
        for size in image_sizes:
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], *size), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], *size), device=crop_logits.device)
            )

        for crop_index, (image_index, start, end, is_tall) in enumerate(origins):
            # Image is tall.
            if is_tall:
                logit_sums[image_index][:, start:end, :] += crop_logits[crop_index]
                logit_counts[image_index][:, start:end, :] += 1
            # Image is wide.
            else:
                logit_sums[image_index][:, :, start:end] += crop_logits[crop_index]
                logit_counts[image_index][:, :, start:end] += 1

        # Average the logits in the regions of overlap.
        return [
            logit_sum / logit_count
            for logit_sum, logit_count in zip(logit_sums, logit_counts)
        ]

    def to_per_pixel_logits_semantic(
        self, mask_logits: Tensor, class_logits: Tensor
    ) -> Tensor:
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            # NOTE(Guarin, 07/25): This is different from the original EoMT code as we
            # keep the logits of the last class whereas EoMT discards them. We discard
            # them later in the `validation_step` function and keep them here for
            # `predict` to work correctly.
            class_logits.softmax(dim=-1),
        )

    def _forward_logits(self, x: Tensor) -> Tensor:
        """Forward pass that returns the logits of the last layer. Intended for
        inference."""
        # x is a batch of images with shape (B, C, H, W).

        # Tiling.
        image_sizes = [img.shape[-2:] for img in x]
        crops_list, origins = self.tile(images=x)
        crops = torch.stack(crops_list)
        crop_h, crop_w = crops.shape[-2:]

        # Forward pass.
        # forward_train returns logits for multiple layers but we only use the last
        # one for inference.
        mask_logits_per_layer, class_logits_per_layer = self.forward_train(crops)
        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]

        # Interpolate and untile.
        mask_logits = F.interpolate(mask_logits, (crop_h, crop_w), mode="bilinear")
        crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
        logits_list = self.untile(
            crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
        )
        logits = torch.stack(logits_list)  # (B, C, H, W)
        return logits

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
