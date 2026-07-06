#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any, Literal

from albumentations import BboxParams
from pydantic import Field

from lightly_train._transforms.ltdetr_transforms.base import (
    LTDETRMosaicArgs,
    LTDETRRandomFlipArgs,
    LTDETRRandomIoUCropArgs,
    LTDETRRandomPhotometricDistortArgs,
    LTDETRRandomZoomOutArgs,
    LTDETRResizeArgs,
)
from lightly_train._transforms.ltdetr_transforms.instance_segmentation import (
    LTDETRInstanceSegmentationTransform,
    LTDETRInstanceSegmentationTransformArgs,
)
from lightly_train._transforms.ltdetr_transforms.utils import (
    ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5,
    ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1,
    resolve_image_size_for_patch_size,
    resolve_ltdetr_step_schedule_for_augmentation,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    MixUpArgs,
    NormalizeArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    ResizeArgs,
)
from lightly_train.types import ImageSizeTuple

logger = logging.getLogger(__name__)


def _resolve_normalize_num_channels(
    normalize: NormalizeArgs, num_channels: int
) -> None:
    if len(normalize.mean) != num_channels:
        logger.debug(
            "Adjusting mean of normalize transform to match num_channels. "
            f"num_channels is {num_channels} but "
            f"normalize.mean has length {len(normalize.mean)}."
        )
        normalize.mean = tuple(
            normalize.mean[i % len(normalize.mean)] for i in range(num_channels)
        )
    if len(normalize.std) != num_channels:
        logger.debug(
            "Adjusting std of normalize transform to match num_channels. "
            f"num_channels is {num_channels} but "
            f"normalize.std has length {len(normalize.std)}."
        )
        normalize.std = tuple(
            normalize.std[i % len(normalize.std)] for i in range(num_channels)
        )


class LTDETRInstanceSegmentationMixUpArgs(MixUpArgs):
    prob: float = 0.5
    step_start: int | Literal["auto"] = "auto"
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRInstanceSegmentationTrainTransformArgs(
    LTDETRInstanceSegmentationTransformArgs
):
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: LTDETRRandomPhotometricDistortArgs | None = Field(
        default_factory=LTDETRRandomPhotometricDistortArgs
    )
    random_zoom_out: LTDETRRandomZoomOutArgs | None = Field(
        default_factory=LTDETRRandomZoomOutArgs
    )
    random_iou_crop: LTDETRRandomIoUCropArgs | None = Field(
        default_factory=LTDETRRandomIoUCropArgs
    )
    random_flip: LTDETRRandomFlipArgs | None = Field(
        default_factory=LTDETRRandomFlipArgs
    )
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(default_factory=LTDETRResizeArgs)
    scale_jitter: None = None
    mosaic: LTDETRMosaicArgs | None = Field(default_factory=LTDETRMosaicArgs)
    mixup: LTDETRInstanceSegmentationMixUpArgs | None = Field(
        default_factory=LTDETRInstanceSegmentationMixUpArgs
    )
    copyblend: None = None
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo",
            label_fields=["class_labels", "indices"],
            min_area=1.0,
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
        patch_size: int | None = model_init_args.get("patch_size")
        if self.image_size == "auto":
            self.image_size = resolve_image_size_for_patch_size(
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
            if normalize is None:
                self.normalize = None
            elif normalize == "none":
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            elif self.normalize is None:
                self.num_channels = 3
            else:
                self.num_channels = len(self.normalize.mean)

        if not isinstance(self.num_channels, int):
            raise RuntimeError("Expected num_channels to be resolved.")
        if isinstance(self.normalize, NormalizeArgs):
            _resolve_normalize_num_channels(self.normalize, self.num_channels)

    def resolve_step_schedule(
        self,
        total_steps: int,
        train_num_batches: int,
        gradient_accumulation_steps: int,
    ) -> None:
        resolve_ltdetr_step_schedule_for_augmentation(
            args=self,
            total_steps=total_steps,
            train_num_batches=train_num_batches,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )


class LTDETRInstanceSegmentationValTransformArgs(
    LTDETRInstanceSegmentationTransformArgs
):
    channel_drop: None = None
    num_channels: int | Literal["auto"] = "auto"
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    resize: ResizeArgs | None = Field(default_factory=LTDETRResizeArgs)
    scale_jitter: None = None
    mosaic: None = None
    mixup: None = None
    copyblend: None = None
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo",
            label_fields=["class_labels", "indices"],
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
        patch_size: int | None = model_init_args.get("patch_size")
        if self.image_size == "auto":
            self.image_size = resolve_image_size_for_patch_size(
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
            if normalize is None:
                self.normalize = None
            elif normalize == "none":
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            elif self.normalize is None:
                self.num_channels = 3
            else:
                self.num_channels = len(self.normalize.mean)

        if not isinstance(self.num_channels, int):
            raise RuntimeError("Expected num_channels to be resolved.")
        if isinstance(self.normalize, NormalizeArgs):
            _resolve_normalize_num_channels(self.normalize, self.num_channels)


class LTDETRInstanceSegmentationTrainTransform(LTDETRInstanceSegmentationTransform):
    transform_args_cls = LTDETRInstanceSegmentationTrainTransformArgs


class LTDETRInstanceSegmentationValTransform(LTDETRInstanceSegmentationTransform):
    transform_args_cls = LTDETRInstanceSegmentationValTransformArgs
