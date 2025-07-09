#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn import functional as F

from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.task_model import TaskModel

logger = logging.getLogger(__name__)


class LinearSegmentationHead(Module):
    """
    Linear segmentation head.
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: patch tokens, shape (B, N, D)
        Returns:
            patch logits, shape (B, N, num_classes)
        """
        logits: Tensor = self.classifier(x)
        return logits


class DINOv2SemanticSegmentation(TaskModel):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        model_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # Pop the backbone weights from model_args if it exists.
        backbone_weights_path = None
        if model_args is not None and "backbone_weights" in model_args:
            backbone_weights_path = model_args.pop("backbone_weights")

        # Disable drop path by default.
        args = {
            "drop_path_rate": 0.0,
        }
        if model_args is not None:
            args.update(model_args)

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
        if backbone_weights_path is not None:
            self.load_backbone_weights(backbone_weights_path)

        # Get the segmentation head.
        self.head = LinearSegmentationHead(embed_dim, num_classes)

    def load_backbone_weights(self, path: str) -> None:
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for inference.

        Args:
            x: input image of shape (B, C, H, W)
        Returns:
            (masks, logits) tuple where masks have shape (B, H, W) and logits have shape
            (B, num_classes, H, W). The masks are the predicted segmentation masks and
            the logits are the raw output of the model.
        """
        # Up-sample to match original image/mask resolution.
        B, _, H, W = x.shape

        # Get the patch tokens -> (B, N, D) where N = H_patch * W_patch.
        patch_tokens = self.backbone(x, is_training=True)["x_norm_patchtokens"]

        # Classify the patch tokens -> (B, N, num_classes).
        logits: Tensor = self.head(patch_tokens)

        # Reshape back to (B, num_classes, H_patch, W_patch).
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        logits = logits.permute(0, 2, 1).reshape(B, -1, H_patch, W_patch)

        # Up-sample to match original image/mask resolution.
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        masks = logits.argmax(dim=1)
        return masks, logits
