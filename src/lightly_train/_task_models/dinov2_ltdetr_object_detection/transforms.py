#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal, Sequence

from albumentations import BboxParams
from pydantic import Field

from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import (
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ScaleJitterArgs,
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


class DINOv2LTDetrObjectDetectionScaleJitterArgs(ScaleJitterArgs):
    sizes: Sequence[tuple[int, int]] | None = [
        (490, 490),
        (518, 518),
        (546, 546),
        (588, 588),
        (616, 616),
        (644, 644),
        (644, 644),
        (644, 644),
        (686, 686),
        (714, 714),
        (742, 742),
        (770, 770),
        (812, 812),
    ]
    min_scale: float | None = 0.76
    max_scale: float | None = 1.27
    num_scales: int | None = 13
    prob: float = 1.0
    # The model is patch 14.
    divisible_by: int | None = 14


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
    # TODO: Lionel (09/25): Remove None, once the stop policy is implemented.
    stop_policy: StopPolicyArgs | None = None
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv2LTDetrObjectDetectionScaleJitterArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )


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
