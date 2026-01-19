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
)

ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5 = RequirementCache("albumentations>=1.4.5")
ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1 = RequirementCache("albumentations>=2.0.1")


class PicoDetRandomPhotometricDistortArgs(RandomPhotometricDistortArgs):
    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)
    prob: float = 0.5


class PicoDetRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5
    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)


class PicoDetRandomIoUCropArgs(RandomIoUCropArgs):
    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000
    prob: float = 0.8


class PicoDetRandomFlipArgs(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class PicoDetScaleJitterArgs(ScaleJitterArgs):
    sizes: Sequence[tuple[int, int]] | None = None
    min_scale: float | None = None
    max_scale: float | None = None
    num_scales: int | None = None
    prob: float = 1.0
    divisible_by: int | None = None
    step_seeding: bool = True
    seed_offset: int = 0


class PicoDetObjectDetectionTrainTransformArgs(ObjectDetectionTransformArgs):
    """PicoDet training transforms aligned with the reference config.

    PicoDet defaults mirror LTDETR training augmentations for consistency.
    """

    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: PicoDetRandomPhotometricDistortArgs | None = Field(
        default_factory=PicoDetRandomPhotometricDistortArgs
    )
    random_zoom_out: PicoDetRandomZoomOutArgs | None = Field(
        default_factory=PicoDetRandomZoomOutArgs
    )
    random_iou_crop: PicoDetRandomIoUCropArgs | None = Field(
        default_factory=PicoDetRandomIoUCropArgs
    )
    random_flip: PicoDetRandomFlipArgs | None = Field(
        default_factory=PicoDetRandomFlipArgs
    )
    image_size: tuple[int, int] | Literal["auto"] = "auto"
    stop_policy: None = None
    resize: ResizeArgs | None = None
    scale_jitter: PicoDetScaleJitterArgs | None = Field(
        default_factory=PicoDetScaleJitterArgs
    )
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
        """Resolve 'auto' values based on model configuration."""
        super().resolve_auto(model_init_args=model_init_args)

        if self.image_size == "auto":
            # Default to 416x416 for PicoDet
            self.image_size = tuple(model_init_args.get("image_size", (416, 416)))

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize", "none")
            if normalize is None:
                self.normalize = None
            elif normalize == "none":
                self.normalize = NormalizeArgs(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = 3

        if self.scale_jitter is not None:
            base = self.image_size[0]
            if base == 416:
                self.scale_jitter.sizes = [
                    (352, 352),
                    (384, 384),
                    (416, 416),
                    (448, 448),
                    (480, 480),
                ]
            elif base == 320:
                self.scale_jitter.sizes = [
                    (256, 256),
                    (288, 288),
                    (320, 320),
                    (352, 352),
                    (384, 384),
                ]
            else:
                self.scale_jitter.sizes = [
                    (base - 64, base - 64),
                    (base - 32, base - 32),
                    (base, base),
                    (base + 32, base + 32),
                    (base + 64, base + 64),
                ]
            self.scale_jitter.divisible_by = 32


class PicoDetObjectDetectionTrainTransform(ObjectDetectionTransform):
    """Training transforms for PicoDet."""

    transform_args_cls = PicoDetObjectDetectionTrainTransformArgs


class PicoDetObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    """PicoDet validation transforms."""

    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    image_size: tuple[int, int] | Literal["auto"] = "auto"
    stop_policy: None = None
    resize: ResizeArgs | None = Field(
        default_factory=lambda: ResizeArgs(height="auto", width="auto")
    )
    scale_jitter: None = None
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
        """Resolve 'auto' values based on model configuration."""
        super().resolve_auto(model_init_args=model_init_args)

        if self.image_size == "auto":
            # Default to 416x416 for PicoDet
            self.image_size = tuple(model_init_args.get("image_size", (416, 416)))

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize", "none")
            if normalize is None:
                self.normalize = None
            elif normalize == "none":
                self.normalize = NormalizeArgs(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = 3


class PicoDetObjectDetectionValTransform(ObjectDetectionTransform):
    """Validation transforms for PicoDet."""

    transform_args_cls = PicoDetObjectDetectionValTransformArgs
