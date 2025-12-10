#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import Field

from lightly_train._transforms.panoptic_segmentation_transform import (
    PanopticSegmentationTransform,
    PanopticSegmentationTransformArgs,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    NormalizeArgs,
    RandomCropArgs,
    RandomFlipArgs,
    ScaleJitterArgs,
    SmallestMaxSizeArgs,
)
from lightly_train.types import ImageSizeTuple


class DINOv3EoMTPanopticSegmentationScaleJitterArgs(ScaleJitterArgs):
    sizes: Sequence[tuple[int, int]] | None = None
    min_scale: float | None = 0.1
    max_scale: float | None = 2.0
    num_scales: int | None = 20
    prob: float = 1.0
    # TODO: Lionel(09/25): This is currently not used.
    divisible_by: int | None = None
    step_seeding: bool = False
    seed_offset: int = 0


class DINOv3EoMTPanopticSegmentationRandomCropArgs(RandomCropArgs):
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"
    pad_if_needed: bool = True
    pad_position: str = "center"
    fill: int = 0
    prob: float = 1.0


class DINOv3EoMTPanopticSegmentationTrainTransformArgs(
    PanopticSegmentationTransformArgs
):
    """
    Defines default transform arguments for panoptic segmentation training with DINOv3.
    """

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    color_jitter: ColorJitterArgs | None = None
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv3EoMTPanopticSegmentationScaleJitterArgs
    )
    smallest_max_size: SmallestMaxSizeArgs | None = None
    random_crop: RandomCropArgs = Field(
        default_factory=DINOv3EoMTPanopticSegmentationRandomCropArgs
    )

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
            normalize = model_init_args.get("image_normalize")
            if normalize is None:
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = len(self.normalize.mean)


class DINOv3EoMTPanopticSegmentationValTransformArgs(PanopticSegmentationTransformArgs):
    """
    Defines default transform arguments for panoptic segmentation validation with DINOv3.
    """

    image_size: ImageSizeTuple | Literal["auto"] | None = None
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_flip: RandomFlipArgs | None = None
    color_jitter: ColorJitterArgs | None = None
    scale_jitter: ScaleJitterArgs | None = None
    smallest_max_size: SmallestMaxSizeArgs | None = None
    random_crop: RandomCropArgs | None = None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)
        if self.image_size == "auto":
            self.image_size = None

        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                assert isinstance(self.image_size, tuple)
                height, width = self.image_size
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize")
            if normalize is None:
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = len(self.normalize.mean)


class DINOv3EoMTPanopticSegmentationTrainTransform(PanopticSegmentationTransform):
    transform_args_cls = DINOv3EoMTPanopticSegmentationTrainTransformArgs


class DINOv3EoMTPanopticSegmentationValTransform(PanopticSegmentationTransform):
    transform_args_cls = DINOv3EoMTPanopticSegmentationValTransformArgs
