#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

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
    min_scale: float = 0.5
    max_scale: float = 2.0
    num_scales: int = 20
    prob: float = 1.0


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

    image_size: tuple[int, int] = (518, 518)
    channel_drop: ChannelDropArgs | None = None
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    random_flip: RandomFlipArgs = Field(default_factory=RandomFlipArgs)
    color_jitter: DINOv2EoMTSemanticSegmentationColorJitterArgs = Field(
        default_factory=DINOv2EoMTSemanticSegmentationColorJitterArgs
    )
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv2EoMTSemanticSegmentationScaleJitterArgs
    )
    smallest_max_size: SmallestMaxSizeArgs | None = None
    random_crop: RandomCropArgs = Field(
        default_factory=DINOv2EoMTSemanticSegmentationRandomCropArgs
    )


class DINOv2EoMTSemanticSegmentationValTransformArgs(SemanticSegmentationTransformArgs):
    """
    Defines default transform arguments for semantic segmentation validation with DINOv2.
    """

    image_size: tuple[int, int] = (518, 518)
    channel_drop: ChannelDropArgs | None = None
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    random_flip: RandomFlipArgs | None = None
    color_jitter: ColorJitterArgs | None = None
    scale_jitter: ScaleJitterArgs | None = None
    smallest_max_size: SmallestMaxSizeArgs = Field(
        default_factory=DINOv2EoMTSemanticSegmentationSmallestMaxSizeArgs
    )
    random_crop: RandomCropArgs | None = None


class DINOv2EoMTSemanticSegmentationTrainTransform(SemanticSegmentationTransform):
    transform_args_cls = DINOv2EoMTSemanticSegmentationTrainTransformArgs


class DINOv2EoMTSemanticSegmentationValTransform(SemanticSegmentationTransform):
    transform_args_cls = DINOv2EoMTSemanticSegmentationValTransformArgs
