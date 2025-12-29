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
from lightly_train._models import package_helpers
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from lightly_train._models.dinov3.dinov3_src.layers.attention import (
    SelfAttention,
)
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.dinov3_eomt_instance_segmentation.scale_block import (
    ScaleBlock,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class DINOv3EoMTInstanceSegmentation(TaskModel):
    model_suffix = "eomt"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]],
        num_queries: int,
        num_joint_blocks: int,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """
        Args:
            model_name:
                The model name. For example "vits14-eomt".
            classes:
                A dict mapping the class ID to the class name. The dict must only
                contain the classes that the model should predict. It must NOT contain
                classes that are in the dataset but should be ignored by the model.
            image_size:
                The size of the input images.
            image_normalize:
                A dict containing the mean and standard deviation for normalizing
                the input images. The dict must contain the keys "mean" and "std".
                Example: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}.
                This is used to normalize the input images before passing them to the
                model.
            num_queries:
                The number of query tokens to use in the model. This is the number of
                individual segments that the model will predict.
            num_joint_blocks:
                The number of blocks that process the query tokens and image tokens
                jointly.
            backbone_weights:
                The path to the DINOv3 backbone weights. The weights must be exported
                using LightlyTrain.
            backbone_args:
                Additional arguments to pass to the DINOv3 backbone.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        super().__init__(locals(), ignore_args={"backbone_weights", "load_weights"})
        parsed_name = self.parse_model_name(model_name=model_name)
        self.model_name = parsed_name["model_name"]
        self.classes = classes
        self.image_size = image_size
        self.image_normalize = image_normalize

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        # An additional class is added to represent "unknown/ignored classes" if needed.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

        # NOTE(Guarin, 08/25): We don't set drop_path_rate=0 here because it is already
        # set by DINOv3.
        backbone_model_args: dict[str, Any] = {
            "in_chans": len(self.image_normalize["mean"]),
        }
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)

        # Get the backbone.
        backbone = DINOV3_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_model_args,
            load_weights=load_weights,
        )
        assert isinstance(backbone, DinoVisionTransformer)
        self.backbone = backbone

        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size

        # TODO(Guarin, 07/25): Improve how mask tokens are handled for fine-tuning.
        # Should we drop them from the model? We disable grads here for DDP to work
        # without find_unused_parameters=True.
        self.backbone.mask_token.requires_grad = False

        # Load the backbone weights if a path is provided.
        # TODO(Thomas,07/2026): this should be done in the package.
        if load_weights and backbone_weights is not None:
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
        self.attn_mask_probs: Tensor
        self.register_buffer(
            "attn_mask_probs", torch.ones(self.num_joint_blocks), persistent=False
        )

        if hasattr(self, "register_load_state_dict_pre_hook"):
            self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                task_model_helpers.queries_adjust_num_queries_hook
            )
            self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                task_model_helpers.class_head_reuse_or_reinit_hook
            )
        else:
            # Backwards compatibility for PyTorch <= 2.4
            self._register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                task_model_helpers.queries_adjust_num_queries_hook, with_module=True
            )
            self._register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                task_model_helpers.class_head_reuse_or_reinit_hook, with_module=True
            )

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}" for name in DINOV3_PACKAGE.list_model_names()
        ]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            cls.parse_model_name(model_name=model)
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def parse_model_name(cls, model_name: str) -> dict[str, str]:
        def raise_invalid_name() -> None:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available "
                f"models are: {cls.list_model_names()}. See the documentation for "
                "more information: https://docs.lightly.ai/train/stable/instance_segmentation.html"
            )

        if not model_name.endswith(f"-{cls.model_suffix}"):
            raise_invalid_name()

        backbone_name = model_name[: -len(f"-{cls.model_suffix}")]

        try:
            package_name, backbone_name = package_helpers.parse_model_name(
                backbone_name
            )
        except ValueError:
            raise_invalid_name()

        if package_name != DINOV3_PACKAGE.name:
            raise_invalid_name()

        try:
            backbone_name = DINOV3_PACKAGE.parse_model_name(model_name=backbone_name)
        except ValueError:
            raise_invalid_name()

        return {
            "model_name": f"{DINOV3_PACKAGE.name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
        }

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.8
    ) -> dict[str, Tensor]:
        """Returns the predicted mask for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape (C, H, W).
            threshold:
                The confidence threshold for the predicted masks. Only masks with a
                confidence score above this threshold are returned.

        Returns:
            A {"labels": Tensor, "masks": Tensor, "scores": Tensor} dict. Labels is a
            tensor of shape (Q,) containing the predicted class for each query. Masks is
            a tensor of shape (Q, H, W) containing the predicted mask for each query.
            Scores is a tensor of shape (Q,) containing the confidence score for each
            query.
        """
        self._track_inference()
        if self.training:
            self.eval()

        # Load image
        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)
        x = transforms_functional.normalize(
            x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
        )

        x, (crop_h, crop_w) = self.resize_and_pad(x)
        x = x.unsqueeze(0)  # (1, C, H', W')

        # (1, Q, H', W'), (1, Q, K+1), Q = num_queries, K = len(self.classes)
        mask_logits, class_logits = self._forward_logits(x)

        # Interpolate to original image size.
        mask_logits = mask_logits[..., :crop_h, :crop_w]  # (1, Q, crop_h, crop_w)
        # (1, Q, H, W)
        mask_logits = F.interpolate(
            mask_logits, size=(image_h, image_w), mode="bilinear"
        )

        # (1, Q), (1, Q, H, W), (1, Q)
        labels, masks, scores = self.get_labels_masks_scores(
            mask_logits=mask_logits, class_logits=class_logits
        )

        # Map internal class IDs to class IDs.
        labels = self.internal_class_to_class[labels]  # (1, Q)

        # Remove batch dimension.
        labels = labels.squeeze(0)
        masks = masks.squeeze(0)
        scores = scores.squeeze(0)

        # Apply threshold.
        keep = scores >= threshold
        labels = labels[keep]
        masks = masks[keep]
        scores = scores[keep]

        return {
            "labels": labels,
            "masks": masks,
            "scores": scores,
        }

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Function used for ONNX export
        # (1, Q, H, W), (1, Q, K+1), Q = num_queries, K = len(self.classes)
        mask_logits, class_logits = self._forward_logits(x)
        labels, masks, scores = self.get_labels_masks_scores(
            mask_logits=mask_logits, class_logits=class_logits
        )

        # Map internal class IDs to class IDs.
        labels = self.internal_class_to_class[labels]
        return labels, masks, scores

    # TODO(Guarin, 07/25): Refactor to take attn_mask_probs as input.
    def forward_train(
        self, x: Tensor, return_logits_per_layer: bool
    ) -> tuple[list[Tensor], list[Tensor]]:
        _, _, H, W = x.shape
        patch_size = self.backbone.patch_size
        num_backbone_blocks = len(self.backbone.blocks)  # type: ignore[arg-type]

        # Match the logic of the PatchEmbded forward
        # (src/lightly_train/_models/dinov3/dinov3_src/layers/patch_embed.py).
        # TODO(Thomas, 09/25): Update the patch embedding logic to not drop extra pixels.
        assert patch_size is not None
        grid_size = (H // patch_size, W // patch_size)

        x, image_size = self.backbone.prepare_tokens_with_masks(x)
        mask_logits_per_layer, class_logits_per_layer = [], []
        for i, block in enumerate(self.backbone.blocks):  # type: ignore[arg-type]
            attn_mask = None

            rope_sincos: tuple[Tensor, Tensor] | None = None
            if self.backbone.rope_embed is not None:
                rope_sincos = self.backbone.rope_embed(H=image_size[0], W=image_size[1])  # type: ignore

            if i == num_backbone_blocks - self.num_joint_blocks:
                # Prepend query tokens.
                x = torch.cat(
                    (self.queries.weight[None, :, :].expand(x.shape[0], -1, -1), x),
                    dim=1,
                )

            if (
                return_logits_per_layer
                and i >= num_backbone_blocks - self.num_joint_blocks
            ):
                mask_logits, class_logits = self._predict(
                    self.backbone.norm(x), grid_size=grid_size
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                # NOTE(Guarin, 08/25): This is different from the original EoMT code.
                # The original code also applies the attention mask during validation.
                # This results is higher reported validation mIoU during training.
                # As attention masking is disabled towards the end of training, the
                # mIoU values converge to the same values whether the attention mask
                # is applied or not. We disable the attention mask as this is also
                # what happens during inference. This way our validation mIoU reflects
                # actual inference performance.
                if self.training:
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
                        # + 1 class token + register tokens
                        self.num_queries + 1 + self.backbone.n_storage_tokens :,
                    ] = interpolated > 0
                    attn_mask = self._disable_attn_mask(
                        attn_mask=attn_mask,
                        prob=self.attn_mask_probs[
                            i - num_backbone_blocks + self.num_joint_blocks
                        ],
                    )

            # TODO(Guarin, 08/25): Double check if sample_drop_ratio > 0 sometimes.
            # This is usually not the case in EoMT but should be verified.
            x = x + block.ls1(  # type: ignore[operator]
                self._attn(block.attn, block.norm1(x), rope=rope_sincos, mask=attn_mask)  # type: ignore
            )
            x = x + block.ls2(block.mlp(block.norm2(x)))  # type: ignore[operator]

        mask_logits, class_logits = self._predict(
            self.backbone.norm(x), grid_size=grid_size
        )
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )

    def _forward_logits(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass that returns the logits of the last layer. Intended for
        inference."""
        # x is a batch of images with shape (B, C, H, W).
        H, W = x.shape[-2:]

        # Forward pass.
        # Only the logits of the last layer are returned.
        mask_logits_per_layer, class_logits_per_layer = self.forward_train(
            x, return_logits_per_layer=False
        )
        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]

        # Interpolate.
        mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
        return mask_logits, class_logits

    def _predict(self, x: Tensor, grid_size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        # TODO(Guarin, 08/25): Investigate if having different norms for queries and
        # patch tokens is beneficial.
        q = x[:, : self.num_queries, :]

        class_logits = self.class_head(q)

        # num queries + 1 class token + register tokens
        x = x[:, self.num_queries + 1 + self.backbone.n_storage_tokens :, :]
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *grid_size)

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.no_grad()
    def get_labels_masks_scores(
        self, mask_logits: Tensor, class_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Get score and label for each query.
        scores = class_logits.softmax(dim=-1)[..., :-1]  # (1, Q, K)
        scores, labels = torch.max(scores, dim=-1)  # (1, Q), (1, Q)

        # Multiply scores with mask scores.
        masks = mask_logits > 0  # (1, Q, H, W)
        # (1, Q)
        mask_scores = (mask_logits.sigmoid().flatten(2) * masks.flatten(2)).sum(2)
        mask_scores = mask_scores / (masks.flatten(2).sum(2) + 1e-6)
        scores = scores * mask_scores  # (1, Q)

        # (1, Q), (1, Q, H, W), (1, Q)
        return labels, masks, scores

    def resize_and_pad(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Resize and pad image to self.image_size while keeping aspect ratio constant.

        Args:
            image:
                A tensor of shape (..., H, W).

        Returns:
            An (image, (crop_h, crop_w)) tuple where image is a tensor of shape
            (..., H', W') with H'==self.image_size[0] and W'==self.image_size[1], and
            (crop_h, crop_w) are the height and width of the resized (non-padded) image.
        """
        image_h, image_w = image.shape[-2:]
        resize_factor = min(self.image_size[0] / image_h, self.image_size[1] / image_w)
        crop_h = round(image_h * resize_factor)
        crop_w = round(image_w * resize_factor)
        pad_h = max(0, self.image_size[0] - crop_h)
        pad_w = max(0, self.image_size[1] - crop_w)
        # (..., crop_h, crop_w)
        image = transforms_functional.resize(image, size=[crop_h, crop_w])
        # (..., H', W')
        image = transforms_functional.pad(image, padding=[0, 0, pad_w, pad_h])
        return image, (crop_h, crop_w)

    # TODO(Guarin, 07/25): No need for attention mask handling in this module. Move it
    # to DINOv3InstanceSegmentationTrain.
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
                self.num_queries + 1 + self.backbone.n_storage_tokens :,
            ][random_queries] = True

        return attn_mask

    # TODO(Guarin, 07/25): Add support for attention masks directly to Attention class?
    def _attn(
        self,
        module: SelfAttention,
        x: Tensor,
        rope: Tensor | tuple[Tensor, Tensor] | None,
        mask: Tensor | None,
    ) -> Tensor:
        # This mirrors DINOv3 Attention forward but with mask support.
        qkv = module.qkv(x)
        B, N, _ = qkv.shape
        C = module.qkv.in_features

        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = module.apply_rope(q, k, rope)
        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2)
        x = x.reshape([B, N, C])
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
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

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
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("model."):
                name = name[len("model.") :]
                new_state_dict[name] = param
        self.load_state_dict(new_state_dict, strict=True)
