#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import Field

from lightly_train._transforms.semantic_segmentation_transform import (
    SemanticSegmentationTransform,
    SemanticSegmentationTransformArgs,
)
from lightly_train._transforms.transform import (
    CenterCropArgs,
    ColorJitterArgs,
    LongestMaxSizeArgs,
    NormalizeArgs,
    RandomCropArgs,
    RandomFlipArgs,
    ScaleJitterArgs,
    SmallestMaxSizeArgs,
)


class DINOv2SemanticSegmentationColorJitterArgs(ColorJitterArgs):
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


class DINOv2SemanticSegmentationScaleJitterArgs(ScaleJitterArgs):
    min_scale: float = 0.5
    max_scale: float = 2.0
    num_scales: int = 20
    prob: float = 1.0


class DINOv2SemanticSegmentationSmallestMaxSizeArgs(SmallestMaxSizeArgs):
    max_size: list[int] = [int(518 * x * 0.1) for x in range(5, 21)]
    prob: float = 1.0


class DINOv2SemanticSegmentationRandomCropArgs(RandomCropArgs):
    height: int = 518
    width: int = 518
    pad_if_needed: bool = True
    pad_position: str = "center"
    fill: int = 0
    fill_mask: int = -100  # Align this value with ignore_index in the loss.
    prob: float = 1.0


class DINOv2SemanticSegmentationCenterCropArgs(CenterCropArgs):
    height: int = 518
    width: int = 518
    pad_if_needed: bool = True
    pad_position: str = "center"
    fill: int = 0
    fill_mask: int = -100  # Align this value with ignore_index in the loss.
    prob: float = 1.0


class DINOv2SemanticSegmentationLongestMaxSizeArgs(LongestMaxSizeArgs):
    max_size: int = 518
    prob: float = 1.0


class DINOv2SemanticSegmentationTrainTransformArgs(SemanticSegmentationTransformArgs):
    """
    Defines default transform arguments for semantic segmentation training with DINOv2.
    """

    ignore_index: int = -100
    image_size: tuple[int, int] = (518, 518)
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    random_flip: RandomFlipArgs = Field(default_factory=RandomFlipArgs)
    color_jitter: DINOv2SemanticSegmentationColorJitterArgs = Field(
        default_factory=DINOv2SemanticSegmentationColorJitterArgs
    )
    scale_jitter: ScaleJitterArgs | None = Field(
        default_factory=DINOv2SemanticSegmentationScaleJitterArgs
    )
    smallest_max_size: SmallestMaxSizeArgs | None = None
    random_crop: RandomCropArgs = Field(
        default_factory=DINOv2SemanticSegmentationRandomCropArgs
    )
    longest_max_size: LongestMaxSizeArgs | None = None
    center_crop: CenterCropArgs | None = None


class DINOv2SemanticSegmentationValTransformArgs(SemanticSegmentationTransformArgs):
    """
    Defines default transform arguments for semantic segmentation validation with DINOv2.
    """

    ignore_index: int = -100
    image_size: tuple[int, int] = (518, 518)
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    random_flip: RandomFlipArgs | None = None
    color_jitter: ColorJitterArgs | None = None
    scale_jitter: ScaleJitterArgs | None = None
    smallest_max_size: SmallestMaxSizeArgs | None = None
    random_crop: RandomCropArgs | None = None
    longest_max_size: LongestMaxSizeArgs = Field(
        default_factory=DINOv2SemanticSegmentationLongestMaxSizeArgs
    )
    center_crop: CenterCropArgs = Field(
        default_factory=DINOv2SemanticSegmentationCenterCropArgs
    )


class DINOv2SemanticSegmentationTrainTransform(SemanticSegmentationTransform):
    def __init__(
        self, transform_args: DINOv2SemanticSegmentationTrainTransformArgs
    ) -> None:
        super().__init__(transform_args=transform_args)

    @staticmethod
    def transform_args_cls() -> type[DINOv2SemanticSegmentationTrainTransformArgs]:
        return DINOv2SemanticSegmentationTrainTransformArgs


class DINOv2SemanticSegmentationValTransform(SemanticSegmentationTransform):
    def __init__(
        self, transform_args: DINOv2SemanticSegmentationValTransformArgs
    ) -> None:
        super().__init__(transform_args=transform_args)

    @staticmethod
    def transform_args_cls() -> type[DINOv2SemanticSegmentationValTransformArgs]:
        return DINOv2SemanticSegmentationValTransformArgs
