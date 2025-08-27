#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import NotRequired

import numpy as np
from albumentations import (
    BasicTransform,
    CenterCrop,
    ColorJitter,
    Compose,
    HorizontalFlip,
    LongestMaxSize,
    Normalize,
    OneOf,
    RandomCrop,
    Resize,
    SmallestMaxSize,
)
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch import Tensor

from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
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


class SemanticSegmentationTransformInput(TaskTransformInput):
    image: NDArray[np.uint8]
    mask: NotRequired[NDArray[np.uint8]]


class SemanticSegmentationTransformOutput(TaskTransformOutput):
    image: Tensor
    mask: NotRequired[Tensor]


class SemanticSegmentationTransformArgs(TaskTransformArgs):
    ignore_index: int
    image_size: tuple[int, int]
    normalize: NormalizeArgs
    random_flip: RandomFlipArgs | None
    color_jitter: ColorJitterArgs | None
    scale_jitter: ScaleJitterArgs | None
    smallest_max_size: SmallestMaxSizeArgs | None
    longest_max_size: LongestMaxSizeArgs | None
    center_crop: CenterCropArgs | None
    random_crop: RandomCropArgs | None


class SemanticSegmentationTransform(TaskTransform):
    transform_args_cls: type[SemanticSegmentationTransformArgs]

    def __init__(self, transform_args: SemanticSegmentationTransformArgs) -> None:
        super().__init__(transform_args)

        # Initialize the list of transforms to apply.
        transform: list[BasicTransform] = []

        if transform_args.scale_jitter is not None:
            # This follows recommendation on how to replace torchvision ScaleJitter with
            # albumentations: https://albumentations.ai/docs/torchvision-kornia2albumentations/
            scales = np.linspace(
                start=transform_args.scale_jitter.min_scale,
                stop=transform_args.scale_jitter.max_scale,
                num=transform_args.scale_jitter.num_scales,
            )
            transform += [
                OneOf(
                    [
                        Resize(
                            height=int(scale * transform_args.image_size[0]),
                            width=int(scale * transform_args.image_size[1]),
                        )
                        for scale in scales
                    ],
                    p=transform_args.scale_jitter.prob,
                )
            ]

        # During training we randomly crop the image to a fixed size
        # without changing the aspect ratio.
        if transform_args.smallest_max_size is not None:
            # Resize the image such that the smallest side is of a fixed size.
            # The aspect ratio is preserved.
            transform += [
                SmallestMaxSize(
                    max_size=transform_args.smallest_max_size.max_size,
                    p=transform_args.smallest_max_size.prob,
                )
            ]

        if transform_args.random_crop is not None:
            transform += [
                RandomCrop(
                    height=transform_args.random_crop.height,
                    width=transform_args.random_crop.width,
                    pad_if_needed=transform_args.random_crop.pad_if_needed,
                    pad_position=transform_args.random_crop.pad_position,
                    fill=transform_args.random_crop.fill,
                    fill_mask=transform_args.ignore_index,
                    p=transform_args.random_crop.prob,
                )
            ]

        # During evaluation we force the image to be of a fixed size
        # using padding if needed. The aspect ratio is preserved and no
        # information is lost if crop size is the same as max_size.
        if transform_args.longest_max_size is not None:
            # Resize the image such that the longest side is of a fixed size.
            transform += [
                LongestMaxSize(
                    max_size=transform_args.longest_max_size.max_size,
                    p=transform_args.longest_max_size.prob,
                )
            ]

            # Center crop the image to a fixed size.
            # No information is lost if crop size is the same as max_size.
            if transform_args.center_crop is None:
                raise ValueError(
                    "center_crop must be provided if longest_max_size is set."
                )

        if transform_args.center_crop is not None:
            transform += [
                CenterCrop(
                    height=transform_args.center_crop.height,
                    width=transform_args.center_crop.width,
                    pad_if_needed=transform_args.center_crop.pad_if_needed,
                    pad_position=transform_args.center_crop.pad_position,
                    fill=transform_args.center_crop.fill,
                    fill_mask=transform_args.ignore_index,
                    p=transform_args.center_crop.prob,
                )
            ]

        # Optionally apply random horizontal flip.
        if transform_args.random_flip is not None:
            transform += [HorizontalFlip(p=transform_args.random_flip.horizontal_prob)]

        # Optionally apply color jitter.
        if transform_args.color_jitter is not None:
            transform += [
                ColorJitter(
                    brightness=transform_args.color_jitter.strength
                    * transform_args.color_jitter.brightness,
                    contrast=transform_args.color_jitter.strength
                    * transform_args.color_jitter.contrast,
                    saturation=transform_args.color_jitter.strength
                    * transform_args.color_jitter.saturation,
                    hue=transform_args.color_jitter.strength
                    * transform_args.color_jitter.hue,
                    p=transform_args.color_jitter.prob,
                )
            ]

        # Normalize the images.
        transform += [
            Normalize(
                mean=transform_args.normalize.mean, std=transform_args.normalize.std
            )
        ]

        # Convert the images to PyTorch tensors.
        transform += [ToTensorV2()]

        # Create the final transform.
        self.transform = Compose(transform, additional_targets={"mask": "mask"})

    def __call__(
        self, input: SemanticSegmentationTransformInput
    ) -> SemanticSegmentationTransformOutput:
        transformed = self.transform(image=input["image"], mask=input["mask"])
        return {"image": transformed["image"], "mask": transformed["mask"]}
