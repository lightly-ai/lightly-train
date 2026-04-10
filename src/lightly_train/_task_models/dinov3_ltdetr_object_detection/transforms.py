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
    ChannelDropArgs,
    CopyBlendArgs,
    MixUpArgs,
    MosaicArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
)
from lightly_train.types import ImageSizeTuple

ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5 = RequirementCache("albumentations>=1.4.5")
ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1 = RequirementCache("albumentations>=2.0.1")


class DINOv3LTDETRObjectDetectionRandomPhotometricDistortArgs(
    RandomPhotometricDistortArgs
):
    prob: float = 0.5

    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)

    # Corresponds to the 4th epoch of the total training run.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_start: int = 25_000
    # Corresponds to the total_epochs - no_aug_epoch of the total training run.
    # None means photometric distort is always on.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 375_000


class DINOv3LTDETRObjectDetectionRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5

    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)

    # Corresponds to the 4th epoch of the total training run.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_start: int = 25_000
    # Corresponds to the total_epochs - no_aug_epoch of the total training run.
    # None means random zoom out is always on.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 375_000


class DINOv3LTDETRObjectDetectionRandomIoUCropArgs(RandomIoUCropArgs):
    prob: float = 0.8

    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000

    # Corresponds to the 4th epoch of the total training run.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_start: int = 25_000
    # Corresponds to the total_epochs - no_aug_epoch of the total training run.
    # None means random IoU crop is always on.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 375_000


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
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
        (640, 640),
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

    # Corresponds to the total_epochs - no_aug_epoch of the total training run.
    # None means scale jitter is always on.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 375_000


class DINOv3LTDETRObjectDetectionMixUpArgs(MixUpArgs):
    prob: float = 0.5

    # Corresponds to the 4th epoch of the total training run.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_start: int = 25_000
    # Corresponds to the 4 + total_epochs // 2 epoch of the total training run.
    # None means mixup is always on.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 250_000


class DINOv3LTDETRObjectDetectionCopyBlendArgs(CopyBlendArgs):
    prob: float = 0.5

    area_threshold: int = 100
    num_objects: int = 3
    expand_ratios: tuple[float, float] = (0.1, 0.25)

    # Corresponds to the 4th epoch of the total training run.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_start: int = 25_000
    # Corresponds to the total_epochs - no_aug_epoch of the total training run.
    # None means copyblend is always on.
    # TODO (Yutong 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 375_000


class DINOv3LTDETRObjectDetectionMosaicArgs(MosaicArgs):
    prob: float = 0.5

    output_size: int = 320
    max_size: int | None = None
    rotation_range: float = 10.0
    translation_range: tuple[float, float] = (0.1, 0.1)
    scaling_range: tuple[float, float] = (0.5, 1.5)
    fill_value: int | float = 0
    max_cached_images: int = 50
    random_pop: bool = True

    # Corresponds to the 4th epoch of the total training run.
    # TODO(Yutong, 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_start: int = 25_000
    # Corresponds to the total_epochs - no_aug_epoch of the total training run.
    # None means mosaic is always on.
    # TODO(Yutong, 04/26): Update step_start and step_stop based on the actual number of training steps.
    step_stop: int | None = 250_000


class DINOv3LTDETRObjectDetectionResizeArgs(ResizeArgs):
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"


class DINOv3LTDETRObjectDetectionTrainTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: ChannelDropArgs | None = None
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
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionResizeArgs
    )
    scale_jitter: DINOv3LTDETRObjectDetectionScaleJitterArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionScaleJitterArgs
    )
    mosaic: DINOv3LTDETRObjectDetectionMosaicArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionMosaicArgs
    )
    mixup: DINOv3LTDETRObjectDetectionMixUpArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionMixUpArgs
    )
    copyblend: DINOv3LTDETRObjectDetectionCopyBlendArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionCopyBlendArgs
    )
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_area=1.0,  # Bbox must have an area of at least 1 pixel.
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
            self.image_size = tuple(model_init_args.get("image_size", (640, 640)))

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


class DINOv3LTDETRObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(
        default_factory=DINOv3LTDETRObjectDetectionResizeArgs
    )
    scale_jitter: ScaleJitterArgs | None = None
    mosaic: DINOv3LTDETRObjectDetectionMosaicArgs | None = None
    mixup: DINOv3LTDETRObjectDetectionMixUpArgs | None = None
    copyblend: DINOv3LTDETRObjectDetectionCopyBlendArgs | None = None
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
            self.image_size = tuple(model_init_args.get("image_size", (640, 640)))

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


class DINOv3LTDETRObjectDetectionTrainTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv3LTDETRObjectDetectionTrainTransformArgs


class DINOv3LTDETRObjectDetectionValTransform(ObjectDetectionTransform):
    transform_args_cls = DINOv3LTDETRObjectDetectionValTransformArgs
