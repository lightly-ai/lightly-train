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
    SmallestMaxSizeArgs,
)


class DINOv2SemanticSegmentationColorJitterArgs(ColorJitterArgs):
    # TODO(Thomas, 07/2025): Adjust these values to match the PhotoMetric Distortion from:
    # https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/transforms/transforms.py#L583
    prob: float = 0.8
    strength: float = 0.5
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.4
    hue: float = 0.2


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

    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    random_flip: RandomFlipArgs = Field(default_factory=RandomFlipArgs)
    color_jitter: DINOv2SemanticSegmentationColorJitterArgs = Field(
        default_factory=DINOv2SemanticSegmentationColorJitterArgs
    )
    smallest_max_size: SmallestMaxSizeArgs = Field(
        default_factory=DINOv2SemanticSegmentationSmallestMaxSizeArgs
    )
    random_crop: RandomCropArgs = Field(
        default_factory=DINOv2SemanticSegmentationRandomCropArgs
    )
    longest_max_size: LongestMaxSizeArgs | None = None
    center_crop: CenterCropArgs | None = None


class DINOv2SemanticSegmentationValTransformArgs(SemanticSegmentationTransformArgs):
    """
    Defines default transform arguments for semantic segmentation validation with DINOv2.
    """

    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    random_flip: RandomFlipArgs | None = None
    color_jitter: ColorJitterArgs | None = None
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
        self, transform_args: DINOv2SemanticSegmentationTrainTransformArgs | None = None
    ) -> None:
        if transform_args is None:
            transform_args = DINOv2SemanticSegmentationTrainTransformArgs()
        super().__init__(transform_args=transform_args)

    @staticmethod
    def transform_args_cls() -> type[DINOv2SemanticSegmentationTrainTransformArgs]:
        return DINOv2SemanticSegmentationTrainTransformArgs


class DINOv2SemanticSegmentationValTransform(SemanticSegmentationTransform):
    def __init__(
        self, transform_args: DINOv2SemanticSegmentationValTransformArgs | None = None
    ) -> None:
        if transform_args is None:
            transform_args = DINOv2SemanticSegmentationValTransformArgs()
        super().__init__(transform_args=transform_args)

    @staticmethod
    def transform_args_cls() -> type[DINOv2SemanticSegmentationValTransformArgs]:
        return DINOv2SemanticSegmentationValTransformArgs
