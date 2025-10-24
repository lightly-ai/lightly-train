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
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
    StopPolicyArgs,
)


class DINOv2LTDETRObjectDetectionRandomPhotometricDistortArgs(
    RandomPhotometricDistortArgs
):
    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)
    prob: float = 0.5


class DINOv2LTDETRObjectDetectionRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5
    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)


class DINOv2LTDETRObjectDetectionRandomIoUCropArgs(RandomIoUCropArgs):
    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000
    prob: float = 0.8


class DINOv2LTDETRObjectDetectionRandomFlipArgs(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class DINOv2LTDETRObjectDetectionScaleJitterArgs(ScaleJitterArgs):
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
    min_scale: float | None = None
    max_scale: float | None = None
    num_scales: int | None = None
    prob: float = 1.0
    divisible_by: int | None = None
    step_seeding: bool = True
    seed_offset: int = 0


class DINOv2LTDETRObjectDetectionResizeArgs(ResizeArgs):
    height: int = 644
    width: int = 644


class DINOv2LTDETRObjectDetectionTrainTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: (
        DINOv2LTDETRObjectDetectionRandomPhotometricDistortArgs | None
    ) = Field(default_factory=DINOv2LTDETRObjectDetectionRandomPhotometricDistortArgs)
    random_zoom_out: DINOv2LTDETRObjectDetectionRandomZoomOutArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionRandomZoomOutArgs
    )
    random_iou_crop: DINOv2LTDETRObjectDetectionRandomIoUCropArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionRandomIoUCropArgs
    )
    random_flip: DINOv2LTDETRObjectDetectionRandomFlipArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionRandomFlipArgs
    )
    image_size: tuple[int, int] = (644, 644)
    # TODO: Lionel (09/25): Remove None, once the stop policy is implemented.
    stop_policy: StopPolicyArgs | None = None
    resize: ResizeArgs | None = None
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionScaleJitterArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )


class DINOv2LTDETRObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    image_size: tuple[int, int] = (644, 644)
    stop_policy: None = None
    resize: ResizeArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionResizeArgs
    )
    scale_jitter: ScaleJitterArgs | None = None
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )


class DINOv2LTDETRObjectDetectionTrainTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDETRObjectDetectionTrainTransformArgs


class DINOv2LTDETRObjectDetectionValTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDETRObjectDetectionValTransformArgs
