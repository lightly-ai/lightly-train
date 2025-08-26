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
from torch.nn import Linear
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class DINOv2LinearSemanticSegmentation(TaskModel):
    model_suffix = "linear"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        class_ignore_index: int | None,
        backbone_freeze: bool,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        image_size: tuple[int, int],
        image_normalize: dict[str, float],
    ) -> None:
        """
        Args:
            model_name:
                The model name. For example "dinov2/vits14-linear".
            classes:
                A dict mapping the class ID to the class name. The dict must only
                contain the classes that the model should predict. It must NOT contain
                classes that are in the dataset but should be ignored by the model.
            class_ignore_index:
                The class ID assigned to pixels that do not belong to any of the
                classes in `classes`. If None, the model will not ignore any classes and
                always assign a class to each pixel.
            backbone_weights:
                The path to the DINOv2 backbone weights. The weights must be exported
                using LightlyTrain.
            backbone_args:
                Additional arguments to pass to the DINOv2 backbone.
            image_size:
                The size to resize images to during inference. Default is (518, 518).
            image_normalize:
                The normalization parameters for images. Default uses ImageNet stats.
        """
        super().__init__(locals(), ignore_args={"backbone_weights"})
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.classes = classes
        self.class_ignore_index = class_ignore_index
        self.backbone_freeze = backbone_freeze
        self.image_size = image_size
        self.image_normalize = image_normalize

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
            model_name=parsed_name["backbone_name"],
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

        if self.backbone_freeze:
            self.freeze_backbone()

        self.head = Linear(embed_dim, len(self.internal_class_to_class))

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}"
            for name in DINOV2_VIT_PACKAGE.list_model_names()
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
                f"models are: {cls.list_model_names()}."
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

        if package_name != DINOV2_VIT_PACKAGE.name:
            raise_invalid_name()

        try:
            backbone_name = DINOV2_VIT_PACKAGE.parse_model_name(
                model_name=backbone_name
            )
        except ValueError:
            raise_invalid_name()

        return {
            "model_name": f"{DINOV2_VIT_PACKAGE.name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
        }

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
        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)
        x = transforms_functional.normalize(
            x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
        )
        # Resize to configured image size
        x = transforms_functional.resize(
            x, size=list(self.image_size)
        )  # (C, H, W) -> (C, H', W')
        x = x.unsqueeze(0)  # (1, C, H', W')

        logits = self._forward_logits(x)  # (1, K|K+1, H', W'), K=num_classes
        # (1, K|K+1, H, W)
        logits = F.interpolate(logits, size=(image_h, image_w), mode="bilinear")

        masks = logits.argmax(dim=1)  # (1, H, W)
        masks = self.internal_class_to_class[masks]  # (1, H, W)
        return masks[0]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Function used for ONNX export
        logits = self._forward_logits(x)  # (B, K, H, W)
        masks = logits.argmax(dim=1)  # (B, H, W)
        masks = self.internal_class_to_class[masks]
        return masks, logits

    def forward_train(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape

        # Get the patch tokens -> (B, N, D) where N = H_patch * W_patch.
        patch_tokens = self.backbone(x, is_training=True)["x_norm_patchtokens"]

        # Classify the patch tokens -> (B, N, K|K+1), K=num_classes
        logits: Tensor = self.head(patch_tokens)

        # Reshape back to (B, K|K+1, H_patch, W_patch).
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        logits = logits.permute(0, 2, 1).reshape(B, -1, H_patch, W_patch)

        # Up-sample to match original image/mask resolution.
        # (B, K|K+1, H, W)
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

        return logits

    # TODO(Guarin, 08/25): Move tile/until as functions to a separate utility module.
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
        crop_logits = self.forward_train(crops)

        # Interpolate and untile.
        crop_logits = F.interpolate(crop_logits, (crop_h, crop_w), mode="bilinear")
        logits_list = self.untile(
            crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
        )
        logits = torch.stack(logits_list)  # (B, C, H, W)
        return logits

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

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
