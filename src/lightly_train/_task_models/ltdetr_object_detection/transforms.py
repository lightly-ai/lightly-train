#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, Literal, Sequence

from albumentations import BboxParams
from lightning_utilities.core.imports import RequirementCache
from pydantic import Field

from lightly_train._task_models.object_detection_components.ltdetr_geometry import (
    ltdetr_image_size_divisor,
)
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
    resolve_ltdetr_step_schedule_for_augmentation,
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


def _resolve_image_size_for_patch_size(
    model_init_args: dict[str, Any],
    *,
    default_image_size: tuple[int, int],
    patch_size: int | None,
) -> tuple[int, int]:
    provided_image_size = model_init_args.get("image_size")
    if provided_image_size is not None:
        image_size = (
            int(provided_image_size[0]),
            int(provided_image_size[1]),
        )
        if patch_size is not None:
            divisor = ltdetr_image_size_divisor(patch_size)
            if any(size % divisor != 0 for size in image_size):
                raise ValueError(
                    "When providing an image size in model_init_args, it must be divisible by 2 * the patch size."
                )
        return image_size

    if patch_size is None:
        return default_image_size

    divisor = ltdetr_image_size_divisor(patch_size)
    return (
        math.ceil(default_image_size[0] / divisor) * divisor,
        math.ceil(default_image_size[1] / divisor) * divisor,
    )


class LTDETRObjectDetectionRandomPhotometricDistortArgs(RandomPhotometricDistortArgs):
    prob: float = 0.5

    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means photometric distort is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5

    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means random zoom out is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionRandomIoUCropArgs(RandomIoUCropArgs):
    prob: float = 0.8

    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means random IoU crop is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionRandomFlipArgs(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class LTDETRObjectDetectionScaleJitterArgs(ScaleJitterArgs):
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
    divisible_by: int | None | Literal["auto"] = "auto"

    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means scale jitter is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionMixUpArgs(MixUpArgs):
    prob: float = 0.5

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" uses a compressed short-run schedule for <= 12 epochs and
    # transitions to the midpoint rule on longer runs.
    # None means mixup is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionCopyBlendArgs(CopyBlendArgs):
    prob: float = 0.5

    area_threshold: int = 100
    num_objects: int = 3
    expand_ratios: tuple[float, float] = (0.1, 0.25)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means copy blend is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionMosaicArgs(MosaicArgs):
    prob: float = 0.5

    output_size: int = 320
    max_size: int | None = None
    rotation_range: float = 10.0
    translation_range: tuple[float, float] = (0.1, 0.1)
    scaling_range: tuple[float, float] = (0.5, 1.5)
    fill_value: int | float = 0
    max_cached_images: int = 50
    random_pop: bool = True

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" uses a compressed short-run schedule for <= 12 epochs and
    # transitions to the midpoint rule on longer runs.
    # None means mosaic is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRObjectDetectionResizeArgs(ResizeArgs):
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"


class LTDETRObjectDetectionTrainTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: LTDETRObjectDetectionRandomPhotometricDistortArgs | None = (
        Field(default_factory=LTDETRObjectDetectionRandomPhotometricDistortArgs)
    )
    random_zoom_out: LTDETRObjectDetectionRandomZoomOutArgs | None = Field(
        default_factory=LTDETRObjectDetectionRandomZoomOutArgs
    )
    random_iou_crop: LTDETRObjectDetectionRandomIoUCropArgs | None = Field(
        default_factory=LTDETRObjectDetectionRandomIoUCropArgs
    )
    random_flip: LTDETRObjectDetectionRandomFlipArgs | None = Field(
        default_factory=LTDETRObjectDetectionRandomFlipArgs
    )
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(default_factory=LTDETRObjectDetectionResizeArgs)
    scale_jitter: LTDETRObjectDetectionScaleJitterArgs | None = Field(
        default_factory=LTDETRObjectDetectionScaleJitterArgs
    )
    mosaic: LTDETRObjectDetectionMosaicArgs | None = Field(
        default_factory=LTDETRObjectDetectionMosaicArgs
    )
    mixup: LTDETRObjectDetectionMixUpArgs | None = Field(
        default_factory=LTDETRObjectDetectionMixUpArgs
    )
    copyblend: LTDETRObjectDetectionCopyBlendArgs | None = Field(
        default_factory=LTDETRObjectDetectionCopyBlendArgs
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

        patch_size: int | None = model_init_args.get("patch_size")

        if self.image_size == "auto":
            self.image_size = _resolve_image_size_for_patch_size(
                model_init_args,
                default_image_size=(640, 640),
                patch_size=patch_size,
            )

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

        if self.scale_jitter is not None:
            if self.scale_jitter.divisible_by == "auto":
                if patch_size is not None:
                    # This is multiplied by 2 to account for the common ViT design. In
                    # our case (05/26) the ViT output a single (H)x(W) scale feature
                    # map and we make a multi-scale from it with the next scales
                    # (H/2)x(W/2), (H)x(W) and (2H)x(2W). That's why we need it to be divisible_by
                    # 2*patch_size, to account for this 2x smaller feature map.
                    # You can take a look at the forward of the DINOv3STAs class.
                    self.scale_jitter.divisible_by = ltdetr_image_size_divisor(
                        patch_size
                    )
                else:
                    self.scale_jitter.divisible_by = None

    def resolve_step_schedule(
        self,
        total_steps: int,
        train_num_batches: int,
        gradient_accumulation_steps: int,
    ) -> None:
        """Resolve ``"auto"`` step_start / step_stop values.

        See :func:`resolve_ltdetr_step_schedule_for_augmentation`.
        """
        resolve_ltdetr_step_schedule_for_augmentation(
            args=self,
            total_steps=total_steps,
            train_num_batches=train_num_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )


class LTDETRObjectDetectionValTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(default_factory=LTDETRObjectDetectionResizeArgs)
    scale_jitter: ScaleJitterArgs | None = None
    mosaic: LTDETRObjectDetectionMosaicArgs | None = None
    mixup: LTDETRObjectDetectionMixUpArgs | None = None
    copyblend: LTDETRObjectDetectionCopyBlendArgs | None = None
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

        patch_size: int | None = model_init_args.get("patch_size")

        if self.image_size == "auto":
            self.image_size = _resolve_image_size_for_patch_size(
                model_init_args,
                default_image_size=(640, 640),
                patch_size=patch_size,
            )

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


class LTDETRObjectDetectionTrainTransform(ObjectDetectionTransform):
    transform_args_cls = LTDETRObjectDetectionTrainTransformArgs


class LTDETRObjectDetectionValTransform(ObjectDetectionTransform):
    transform_args_cls = LTDETRObjectDetectionValTransformArgs


# TODO (Lionel, 06/26): Remove all the `v2` naming once the DINOv2 LT-DETR models are
# completely migrated to the generic LTDETR pipeline.
class DINOv2LTDETRObjectDetectionRandomPhotometricDistortArgsV2(
    RandomPhotometricDistortArgs
):
    prob: float = 0.5

    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means photometric distort is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionRandomZoomOutArgsV2(RandomZoomOutArgs):
    prob: float = 0.5

    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means random zoom out is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionRandomIoUCropArgsV2(RandomIoUCropArgs):
    prob: float = 0.8

    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means random IoU crop is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionRandomFlipArgsV2(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class DINOv2LTDETRObjectDetectionScaleJitterArgsV2(ScaleJitterArgs):
    # Sizes must be multiples of patch size * 2
    sizes: Sequence[tuple[int, int]] | None = [
        (476, 476),
        (504, 504),
        (532, 532),
        (560, 560),
        (588, 588),
        (616, 616),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (644, 644),
        (672, 672),
        (700, 700),
        (728, 728),
        (756, 756),
        (784, 784),
        (812, 812),
    ]
    min_scale: float | None = None
    max_scale: float | None = None
    num_scales: int | None = None
    prob: float = 1.0
    divisible_by: int | None = None

    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means scale jitter is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionMixUpArgsV2(MixUpArgs):
    prob: float = 0.5

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" uses a compressed short-run schedule for <= 12 epochs and
    # transitions to the midpoint rule on longer runs.
    # None means mixup is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionCopyBlendArgsV2(CopyBlendArgs):
    prob: float = 0.5

    area_threshold: int = 100
    num_objects: int = 3
    expand_ratios: tuple[float, float] = (0.1, 0.25)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means copy blend is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionMosaicArgsV2(MosaicArgs):
    prob: float = 0.5

    output_size: int = 320
    max_size: int | None = None
    rotation_range: float = 10.0
    translation_range: tuple[float, float] = (0.1, 0.1)
    scaling_range: tuple[float, float] = (0.5, 1.5)
    fill_value: int | float = 0
    max_cached_images: int = 50
    random_pop: bool = True

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" uses a compressed short-run schedule for <= 12 epochs and
    # transitions to the midpoint rule on longer runs.
    # None means mosaic is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class DINOv2LTDETRObjectDetectionResizeArgsV2(ResizeArgs):
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"


class DINOv2LTDETRObjectDetectionTrainTransformArgsV2(ObjectDetectionTransformArgs):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: (
        DINOv2LTDETRObjectDetectionRandomPhotometricDistortArgsV2 | None
    ) = Field(default_factory=DINOv2LTDETRObjectDetectionRandomPhotometricDistortArgsV2)
    random_zoom_out: DINOv2LTDETRObjectDetectionRandomZoomOutArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionRandomZoomOutArgsV2
    )
    random_iou_crop: DINOv2LTDETRObjectDetectionRandomIoUCropArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionRandomIoUCropArgsV2
    )
    random_flip: DINOv2LTDETRObjectDetectionRandomFlipArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionRandomFlipArgsV2
    )
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionResizeArgsV2
    )
    scale_jitter: DINOv2LTDETRObjectDetectionScaleJitterArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionScaleJitterArgsV2
    )
    mosaic: DINOv2LTDETRObjectDetectionMosaicArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionMosaicArgsV2
    )
    mixup: DINOv2LTDETRObjectDetectionMixUpArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionMixUpArgsV2
    )
    copyblend: DINOv2LTDETRObjectDetectionCopyBlendArgsV2 | None = Field(
        default_factory=DINOv2LTDETRObjectDetectionCopyBlendArgsV2
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

    def resolve_step_schedule(
        self,
        total_steps: int,
        train_num_batches: int,
        gradient_accumulation_steps: int,
    ) -> None:
        """Resolve ``"auto"`` step_start / step_stop values.

        See :func:`resolve_ltdetr_step_schedule_for_augmentation`.
        """
        resolve_ltdetr_step_schedule_for_augmentation(
            args=self,
            total_steps=total_steps,
            train_num_batches=train_num_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )


class DINOv2LTDETRObjectDetectionValTransformArgsV2(ObjectDetectionTransformArgs):
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
        default_factory=DINOv2LTDETRObjectDetectionResizeArgsV2
    )
    scale_jitter: ScaleJitterArgs | None = None
    mosaic: DINOv2LTDETRObjectDetectionMosaicArgsV2 | None = None
    mixup: DINOv2LTDETRObjectDetectionMixUpArgsV2 | None = None
    copyblend: DINOv2LTDETRObjectDetectionCopyBlendArgsV2 | None = None
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


class DINOv2LTDETRObjectDetectionTrainTransformV2(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDETRObjectDetectionTrainTransformArgsV2


class DINOv2LTDETRObjectDetectionValTransformV2(ObjectDetectionTransform):
    transform_args_cls = DINOv2LTDETRObjectDetectionValTransformArgsV2
