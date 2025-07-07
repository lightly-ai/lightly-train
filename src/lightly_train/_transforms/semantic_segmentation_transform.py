#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from albumentations import (
    BasicTransform,
    CenterCrop,
    ColorJitter,
    Compose,
    HorizontalFlip,
    LongestMaxSize,
    Normalize,
    RandomCrop,
    SmallestMaxSize,
)
from albumentations.pytorch import ToTensorV2

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
    SmallestMaxSizeArgs,
)


class SemanticSegmentationTransformArgs(TaskTransformArgs):
    normalize: NormalizeArgs
    random_flip: RandomFlipArgs | None
    color_jitter: ColorJitterArgs | None
    smallest_max_size: SmallestMaxSizeArgs | None
    longest_max_size: LongestMaxSizeArgs | None
    center_crop: CenterCropArgs | None
    random_crop: RandomCropArgs | None


class SemanticSegmentationTransform(TaskTransform):
    def __init__(self, transform_args: SemanticSegmentationTransformArgs) -> None:
        super().__init__(transform_args)

        # Initialize the list of transforms to apply.
        transform: list[BasicTransform] = []

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

            # Randomly crop the image to a fixed size.
            if transform_args.random_crop is None:
                raise ValueError(
                    "random_crop must be provided if smallest_max_size is set."
                )
            transform += [
                RandomCrop(
                    height=transform_args.random_crop.height,
                    width=transform_args.random_crop.width,
                    pad_if_needed=transform_args.random_crop.pad_if_needed,
                    pad_position=transform_args.random_crop.pad_position,
                    fill=transform_args.random_crop.fill,
                    mask_value=transform_args.random_crop.fill_mask,
                    p=transform_args.random_crop.prob,
                )
            ]

        # During evaluation we force the image to be of a fixed size
        # using padding if needed. The aspect ratio is preserved and no
        # information is lost if crop size is the same as max_size.
        elif transform_args.longest_max_size is not None:
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
            transform += [
                CenterCrop(
                    height=transform_args.center_crop.height,
                    width=transform_args.center_crop.width,
                    pad_if_needed=transform_args.center_crop.pad_if_needed,
                    pad_position=transform_args.center_crop.pad_position,
                    fill=transform_args.center_crop.fill,
                    mask_value=transform_args.center_crop.fill_mask,
                    p=transform_args.center_crop.prob,
                )
            ]
        else:
            raise ValueError(
                "Either smallest_max_size or longest_max_size must be "
                "provided in the transform arguments."
            )

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

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        transformed = self.transform(image=input["image"], mask=input["mask"])
        return {"image": transformed["image"], "mask": transformed["mask"]}
