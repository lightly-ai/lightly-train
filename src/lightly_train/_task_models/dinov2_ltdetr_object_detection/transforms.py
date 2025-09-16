#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal, Set

from albumentations import BasicTransform, BboxParams
from pydantic import ConfigDict, Field

from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.random_photometric_distort import (
    RandomPhotometricDistort,
)
from lightly_train._transforms.random_zoom_out import RandomZoomOut
from lightly_train._transforms.transform import (
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    StopPolicyArgs,
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


class DINOv2LTDetrObjectDetectionStopPolicyArgs(StopPolicyArgs):
    stop_step: int = 71
    ops: Set[type[BasicTransform]] = Field(
        default_factory=lambda: {
            RandomPhotometricDistort,
            RandomZoomOut,
            # TODO: Lionel (09/25): Add RandomIoUCrop.
        }
    )


class DINOv2LTDetrObjectDetectionTrainTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: (
        DINOv2LTDetrObjectDetectionRandomPhotometricDistortArgs | None
    ) = Field(default_factory=DINOv2LTDetrObjectDetectionRandomPhotometricDistortArgs)
    random_zoom_out: DINOv2LTDetrObjectDetectionRandomZoomOutArgs | None = Field(
        default_factory=DINOv2LTDetrObjectDetectionRandomZoomOutArgs
    )
    random_flip: DINOv2LTDetrObjectDetectionRandomFlipArgs | None = Field(
        default_factory=DINOv2LTDetrObjectDetectionRandomFlipArgs
    )
    image_size: tuple[int, int] = (644, 644)
    stop_policy: DINOv2LTDetrObjectDetectionStopPolicyArgs | None = Field(
        default_factory=DINOv2LTDetrObjectDetectionStopPolicyArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DINOv2LTDetrObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_flip: None = None
    image_size: tuple[int, int] = (644, 644)
    stop_policy: None = None
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )


class DINOv2LTDetrObjectDetectionTrainTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDetrObjectDetectionTrainTransformArgs


class DINOv2LTDetrObjectDetectionValTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDetrObjectDetectionValTransformArgs
