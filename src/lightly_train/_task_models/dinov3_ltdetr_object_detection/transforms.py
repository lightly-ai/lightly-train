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


class DINOv3LTDETRObjectDetectionRandomPhotometricDistortArgs(
    RandomPhotometricDistortArgs
):
    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)
    prob: float = 0.5


class DINOv3LTDETRObjectDetectionRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5
    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)


class DINOv3LTDETRObjectDetectionRandomIoUCropArgs(RandomIoUCropArgs):
    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000
    prob: float = 0.8


class DINOv3LTDETRObjectDetectionRandomFlipArgs(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class DINOv3LTDETRObjectDetectionScaleJitterArgs(ScaleJitterArgs):
    sizes: Sequence[tuple[int, int]] | None = [
        (480, 480),
        (512, 512),
        (544, 544),
        (576, 576),
        (608, 608),
        (640, 640),
        (640, 640),
        (640, 640),
        (672, 672),
        (704, 704),
        (736, 736),
        (768, 768),
        (800, 800),
    ]
    min_scale: float | None = None
    max_scale: float | None = None
    num_scales: int | None = None
    prob: float = 1.0
    divisible_by: int | None = None
    step_seeding: bool = True
    seed_offset: int = 0


class DINOv3LTDETRObjectDetectionResizeArgs(ResizeArgs):
    height: int = 640
    width: int = 640


class DINOv3LTDETRObjectDetectionTrainTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: (
        DINOv3LTDETRObjectDetectionRandomPhotometricDistortArgs | None
    ) = Field(default_factory=DINOv3LTDETRObjectDetectionRandomPhotometricDistortArgs)
    random_zoom_out: DINOv3LTDETRObjectDetectionRandomZoomOutArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionRandomZoomOutArgs
    )
    random_iou_crop: DINOv3LTDETRObjectDetectionRandomIoUCropArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionRandomIoUCropArgs
    )
    random_flip: DINOv3LTDETRObjectDetectionRandomFlipArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionRandomFlipArgs
    )
    image_size: tuple[int, int] = (640, 640)
    # TODO: Lionel (09/25): Remove None, once the stop policy is implemented.
    stop_policy: StopPolicyArgs | None = None
    resize: ResizeArgs | None = None
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionScaleJitterArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )


class DINOv3LTDETRObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    image_size: tuple[int, int] = (640, 640)
    stop_policy: None = None
    resize: ResizeArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionResizeArgs
    )
    scale_jitter: ScaleJitterArgs | None = None
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"], min_width=0.0, min_height=0.0
        ),
    )


class DINOv3LTDETRObjectDetectionTrainTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv3LTDETRObjectDetectionTrainTransformArgs


class DINOv3LTDETRObjectDetectionValTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv3LTDETRObjectDetectionValTransformArgs
