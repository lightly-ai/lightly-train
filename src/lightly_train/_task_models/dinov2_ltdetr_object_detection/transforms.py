#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal, Sequence

from albumentations import BboxParams
from lightning_utilities.core.imports import RequirementCache
from pydantic import Field

from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import (
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
    StopPolicyArgs,
)
from lightly_train.types import ImageSizeTuple

ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5 = RequirementCache("albumentations>=1.4.5")
ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1 = RequirementCache("albumentations>=2.0.1")


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
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"


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
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    # TODO: Lionel (09/25): Remove None, once the stop policy is implemented.
    stop_policy: StopPolicyArgs | None = None
    resize: ResizeArgs | None = None
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionScaleJitterArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_width=0.0,
            min_height=0.0,
            **(
                dict(filter_invalid_bboxes=True)
                if ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1
                else {}
            ),
            **(dict(clip=True) if ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5 else {}),
        ),
    )
    normalize: NormalizeArgs | Literal["auto"] | None = "auto"

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)

        if self.image_size == "auto":
            self.image_size = tuple(model_init_args.get("image_size", (644, 644)))

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize", "none")
            # Normalize is specifically set to None in model_init_args.
            if normalize is None:
                self.normalize = None
            # Normalize is not set in model_init_args.
            elif normalize == "none":
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                if self.normalize is None:
                    self.num_channels = 3
                else:
                    self.num_channels = len(self.normalize.mean)


class DINOv2LTDETRObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    stop_policy: None = None
    resize: ResizeArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionResizeArgs
    )
    scale_jitter: ScaleJitterArgs | None = None
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_width=0.0,
            min_height=0.0,
            **(
                dict(filter_invalid_bboxes=True)
                if ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1
                else {}
            ),
            **(dict(clip=True) if ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5 else {}),
        ),
    )
    normalize: NormalizeArgs | Literal["auto"] | None = "auto"

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)

        if self.image_size == "auto":
            self.image_size = tuple(model_init_args.get("image_size", (644, 644)))

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize", "none")
            # Normalize is specifically set to None in model_init_args.
            if normalize is None:
                self.normalize = None
            # Normalize is not set in model_init_args.
            elif normalize == "none":
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                if self.normalize is None:
                    self.num_channels = 3
                else:
                    self.num_channels = len(self.normalize.mean)


class DINOv2LTDETRObjectDetectionTrainTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDETRObjectDetectionTrainTransformArgs


class DINOv2LTDETRObjectDetectionValTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDETRObjectDetectionValTransformArgs
