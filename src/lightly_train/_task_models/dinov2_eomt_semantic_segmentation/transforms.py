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

from lightly_train._transforms.semantic_segmentation_transform import (
    SemanticSegmentationTransform,
    SemanticSegmentationTransformArgs,
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


class DINOv2EoMTSemanticSegmentationColorJitterArgs(ColorJitterArgs):
    # Differences between EoMT and this transform:
    # - EoMT always applies brightness before contrast/saturation/hue.
    # - EoMT applies all transforms indedenently with probability 0.5. We apply either
    #   all or none with probability 0.5.
    prob: float = 0.5
    strength: float = 1.0
    brightness: float = 32.0 / 255.0
    contrast: float = 0.5
    saturation: float = 0.5
    hue: float = 18.0 / 360.0


class DINOv2EoMTSemanticSegmentationScaleJitterArgs(ScaleJitterArgs):
    sizes: Sequence[tuple[int, int]] | None = None
    min_scale: float | None = 0.5
    max_scale: float | None = 2.0
    num_scales: int | None = 20
    prob: float = 1.0
    # TODO: Lionel(09/25): This is currently not used.
    divisible_by: int | None = None
    step_seeding: bool = False
    seed_offset: int = 0


class DINOv2EoMTSemanticSegmentationSmallestMaxSizeArgs(SmallestMaxSizeArgs):
    max_size: int | list[int] | Literal["auto"] = "auto"
    prob: float = 1.0


class DINOv2EoMTSemanticSegmentationRandomCropArgs(RandomCropArgs):
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"
    pad_if_needed: bool = True
    pad_position: str = "center"
    fill: int = 0
    prob: float = 1.0


class DINOv2EoMTSemanticSegmentationTrainTransformArgs(
    SemanticSegmentationTransformArgs
):
    """
    Defines default transform arguments for semantic segmentation training with DINOv2.
    """

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    color_jitter: DINOv2EoMTSemanticSegmentationColorJitterArgs | None = Field(
        default_factory=DINOv2EoMTSemanticSegmentationColorJitterArgs
    )
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv2EoMTSemanticSegmentationScaleJitterArgs
    )
    smallest_max_size: SmallestMaxSizeArgs | None = None
    random_crop: RandomCropArgs = Field(
        default_factory=DINOv2EoMTSemanticSegmentationRandomCropArgs
    )

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)
        if self.image_size == "auto":
            self.image_size = tuple(model_init_args.get("image_size", (518, 518)))

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


class DINOv2EoMTSemanticSegmentationValTransformArgs(SemanticSegmentationTransformArgs):
    """
    Defines default transform arguments for semantic segmentation validation with DINOv2.
    """

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_flip: RandomFlipArgs | None = None
    color_jitter: ColorJitterArgs | None = None
    scale_jitter: ScaleJitterArgs | None = None
    smallest_max_size: SmallestMaxSizeArgs = Field(
        default_factory=DINOv2EoMTSemanticSegmentationSmallestMaxSizeArgs
    )
    random_crop: RandomCropArgs | None = None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)
        if self.image_size == "auto":
            self.image_size = tuple(model_init_args.get("image_size", (518, 518)))

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


class DINOv2EoMTSemanticSegmentationTrainTransform(SemanticSegmentationTransform):
    transform_args_cls = DINOv2EoMTSemanticSegmentationTrainTransformArgs


class DINOv2EoMTSemanticSegmentationValTransform(SemanticSegmentationTransform):
    transform_args_cls = DINOv2EoMTSemanticSegmentationValTransformArgs
