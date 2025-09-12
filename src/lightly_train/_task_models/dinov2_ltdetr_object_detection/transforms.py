#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from albumentations import BboxParams
from pydantic import Field

from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import (
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ResizeArgs,
)


class DINOv2LTDetrObjectDetectionRandomPhotometricDistortArgs(
    RandomPhotometricDistortArgs
):
    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)
    prob: float = 0.5


class DINOv2LTDetrObjectDetectionRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5
    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)


class DINOv2LTDetrObjectDetectionRandomFlipArgs(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class DINOv2LTDetrObjectDetectionResizeArgs(ResizeArgs):
    height: int = 644
    width: int = 644


class DINOv2LTDetrObjectDetectionTransformArgs(ObjectDetectionTransformArgs):
    photometric_distort: DINOv2LTDetrObjectDetectionRandomPhotometricDistortArgs = (
        Field(default_factory=DINOv2LTDetrObjectDetectionRandomPhotometricDistortArgs)
    )
    random_zoom_out: DINOv2LTDetrObjectDetectionRandomZoomOutArgs = Field(
        default_factory=DINOv2LTDetrObjectDetectionRandomZoomOutArgs
    )
    random_flip: DINOv2LTDetrObjectDetectionRandomFlipArgs = Field(
        default_factory=DINOv2LTDetrObjectDetectionRandomFlipArgs
    )
    resize: DINOv2LTDetrObjectDetectionResizeArgs = Field(
        default_factory=DINOv2LTDetrObjectDetectionResizeArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )
