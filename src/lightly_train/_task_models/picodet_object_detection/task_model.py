#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._data import file_helpers
from lightly_train._task_models.picodet_object_detection.csp_pan import CSPPAN
from lightly_train._task_models.picodet_object_detection.esnet import ESNet
from lightly_train._task_models.picodet_object_detection.pico_head import PicoHead
from lightly_train._task_models.picodet_object_detection.postprocessor import (
    PicoDetPostProcessor,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

# Model configurations
_MODEL_CONFIGS = {
    "picodet/s-320": {
        "model_size": "s",
        "image_size": (320, 320),
        "stacked_convs": 2,
    },
    "picodet/s-416": {
        "model_size": "s",
        "image_size": (416, 416),
        "stacked_convs": 2,
    },
}


class PicoDetObjectDetection(TaskModel):
    """PicoDet-S object detection model.

    PicoDet is a lightweight anchor-free object detector designed for
    mobile and edge deployment. It uses an Enhanced ShuffleNet backbone,
    CSP-PAN neck, and GFL-style detection head.

    Supported models:
        - picodet/s-320: PicoDet-S at 320×320 input
        - picodet/s-416: PicoDet-S at 416×416 input

    Args:
        model_name: Model variant name.
        image_size: Input image size (H, W).
        num_classes: Number of object classes.
        image_normalize: Normalization parameters (mean, std in [0,1] space).
        reg_max: Maximum value for DFL distribution.
        score_threshold: Default score threshold for inference.
        iou_threshold: IoU threshold for NMS.
        max_detections: Maximum number of detections.
        load_weights: Whether to load pretrained weights (unused for scratch).
    """

    model_suffix = "picodet"

    def __init__(
        self,
        *,
        model_name: str,
        image_size: tuple[int, int],
        num_classes: int,
        image_normalize: dict[str, list[float]] | None = None,
        reg_max: int = 7,
        score_threshold: float = 0.025,
        iou_threshold: float = 0.6,
        max_detections: int = 100,
        load_weights: bool = True,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights"})

        self.model_name = model_name
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.num_classes = num_classes
        self.reg_max = reg_max

        # Get model config
        config = _MODEL_CONFIGS.get(model_name)
        if config is None:
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                f"Available: {list(_MODEL_CONFIGS.keys())}"
            )

        model_size_raw = config["model_size"]
        stacked_convs_raw = config["stacked_convs"]
        if model_size_raw not in ("s", "m", "l"):
            raise ValueError(f"Invalid model_size: {model_size_raw}")
        if not isinstance(stacked_convs_raw, int):
            raise TypeError(f"stacked_convs must be int, got {type(stacked_convs_raw)}")
        model_size_typed: Literal["s", "m", "l"] = model_size_raw  # type: ignore[assignment]
        stacked_convs_typed: int = stacked_convs_raw

        # Build backbone
        self.backbone = ESNet(
            model_size=model_size_typed,
            out_indices=(2, 9, 12),  # C3, C4, C5
        )
        backbone_out_channels = self.backbone.out_channels

        # Build neck
        self.neck = CSPPAN(
            in_channels=backbone_out_channels,
            out_channels=96,
            kernel_size=5,
            num_features=4,  # P3, P4, P5, P6
            expansion=1.0,
            num_csp_blocks=1,
            use_depthwise=True,
        )

        # Build head
        self.head = PicoHead(
            in_channels=96,
            num_classes=num_classes,
            feat_channels=96,
            stacked_convs=stacked_convs_typed,
            kernel_size=5,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            share_cls_reg=True,
            use_depthwise=True,
        )

        # Build postprocessor
        self.postprocessor = PicoDetPostProcessor(
            num_classes=num_classes,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    @classmethod
    def list_model_names(cls) -> list[str]:
        """Return list of supported model names."""
        return list(_MODEL_CONFIGS.keys())

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        """Check if a model name is supported."""
        return model in _MODEL_CONFIGS

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load the state dict from a training checkpoint.

        Loads EMA weights if available, otherwise falls back to model weights.

        Args:
            state_dict: Checkpoint state dict.
            strict: Whether to strictly enforce key matching.
            assign: Whether to assign parameters instead of copying.

        Returns:
            Incompatible keys from loading.
        """
        has_ema_weights = any(k.startswith("ema_model.model.") for k in state_dict)
        has_model_weights = any(k.startswith("model.") for k in state_dict)

        new_state_dict = {}
        if has_ema_weights:
            # Prefer EMA weights
            for name, param in state_dict.items():
                if name.startswith("ema_model.model."):
                    new_name = name[len("ema_model.model.") :]
                    new_state_dict[new_name] = param
        elif has_model_weights:
            # Fall back to regular model weights
            for name, param in state_dict.items():
                if name.startswith("model."):
                    new_name = name[len("model.") :]
                    new_state_dict[new_name] = param
        else:
            # Assume it's already a task model state dict
            new_state_dict = state_dict

        return self.load_state_dict(new_state_dict, strict=strict, assign=assign)

    def deploy(self) -> Self:
        """Set the model to deployment mode."""
        self.eval()
        self.postprocessor.deploy()
        return self

    def forward(self, images: Tensor) -> dict[str, list[Tensor]]:
        """Forward pass returning raw per-level predictions.

        Args:
            images: Input tensor of shape (B, C, H, W).

        Returns:
            Dictionary with:
            - cls_scores: List of (B, num_classes, H, W) per level.
            - bbox_preds: List of (B, 4*(reg_max+1), H, W) per level.
        """
        feats = self.backbone(images)
        feats = self.neck(feats)
        cls_scores, bbox_preds = self.head(feats)
        return {"cls_scores": cls_scores, "bbox_preds": bbox_preds}

    @torch.no_grad()
    def predict(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.6,
    ) -> dict[str, Tensor]:
        """Run inference on a single image.

        Args:
            image: Input image as path, PIL image, or tensor (C, H, W).
            threshold: Score threshold for detections.

        Returns:
            Dictionary with:
            - labels: Tensor of shape (N,) with class indices.
            - bboxes: Tensor of shape (N, 4) with boxes in xyxy format.
            - scores: Tensor of shape (N,) with confidence scores.
        """
        self._track_inference()
        self.eval()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = x.shape[-2:]

        # Preprocess
        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)
        if self.image_normalize is not None:
            x = transforms_functional.normalize(
                x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
            )
        x = transforms_functional.resize(x, list(self.image_size))
        x = x.unsqueeze(0)

        # Forward
        outputs = self(x)

        # Postprocess
        results: dict[str, Tensor] = self.postprocessor(
            cls_scores=outputs["cls_scores"],
            bbox_preds=outputs["bbox_preds"],
            original_size=(orig_h, orig_w),
            score_threshold=threshold,
        )

        return results
