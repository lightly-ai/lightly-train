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

import numpy as np
from albumentations import (
    BasicTransform,
    ColorJitter,
    Compose,
    HorizontalFlip,
    OneOf,
    RandomCrop,
    Resize,
    SmallestMaxSize,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
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
from lightly_train.types import ImageSizeTuple, NDArrayImage, NDArrayMask

logger = logging.getLogger(__name__)


class SemanticSegmentationTransformInput(TaskTransformInput):
    image: NDArrayImage
    mask: NotRequired[NDArrayMask]


class SemanticSegmentationTransformOutput(TaskTransformOutput):
    image: Tensor
    mask: NotRequired[Tensor]


class SemanticSegmentationTransformArgs(TaskTransformArgs):
    ignore_index: int
    image_size: ImageSizeTuple | Literal["auto"]
    channel_drop: ChannelDropArgs | None
    num_channels: int | Literal["auto"]
    normalize: NormalizeArgs | Literal["auto"]
    random_flip: RandomFlipArgs | None
    color_jitter: ColorJitterArgs | None
    # TODO: Lionel(09/25): These are currently not fully used.
    scale_jitter: ScaleJitterArgs | None
    smallest_max_size: SmallestMaxSizeArgs | None
    random_crop: RandomCropArgs | None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        pass

    def resolve_incompatible(self) -> None:
        self.normalize = no_auto(self.normalize)
        assert isinstance(self.normalize, NormalizeArgs)
        num_channels = no_auto(self.num_channels)
        assert isinstance(num_channels, int)

        # Adjust normalization mean and std to match num_channels.
        if len(self.normalize.mean) != num_channels:
            logger.debug(
                "Adjusting mean of normalize transform to match num_channels. "
                f"num_channels is {num_channels} but "
                f"normalize.mean has length {len(self.normalize.mean)}."
            )
            # Repeat the values until they match num_channels.
            self.normalize.mean = tuple(
                self.normalize.mean[i % len(self.normalize.mean)]
                for i in range(num_channels)
            )
        if len(self.normalize.std) != num_channels:
            logger.debug(
                "Adjusting std of normalize transform to match num_channels. "
                f"num_channels is {num_channels} but "
                f"normalize.std has length {len(self.normalize.std)}."
            )
            # Repeat the values until they match num_channels.
            self.normalize.std = tuple(
                self.normalize.std[i % len(self.normalize.std)]
                for i in range(num_channels)
            )

        # Disable color jitter if necessary.
        if self.color_jitter is not None and num_channels != 3:
            logger.debug(
                "Disabling color jitter transform as it only supports 3-channel "
                f"images but num_channels is {num_channels}."
            )
            self.color_jitter = None


class SemanticSegmentationTransform(TaskTransform):
    transform_args_cls: type[SemanticSegmentationTransformArgs] = (
        SemanticSegmentationTransformArgs
    )

    def __init__(
        self,
        transform_args: SemanticSegmentationTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        # Initialize the list of transforms to apply.
        transform: list[BasicTransform] = []

        if transform_args.channel_drop is not None:
            transform += [
                ChannelDrop(
                    num_channels_keep=transform_args.channel_drop.num_channels_keep,
                    weight_drop=transform_args.channel_drop.weight_drop,
                )
            ]

        if transform_args.scale_jitter is not None:
            # TODO (Lionel, 09/25): Use our custom ScaleJitter transform.

            # This follows recommendation on how to replace torchvision ScaleJitter with
            # albumentations: https://albumentations.ai/docs/torchvision-kornia2albumentations/
            assert transform_args.scale_jitter.min_scale is not None
            assert transform_args.scale_jitter.max_scale is not None
            assert transform_args.scale_jitter.num_scales is not None
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
                    max_size=no_auto(transform_args.smallest_max_size.max_size),
                    p=transform_args.smallest_max_size.prob,
                )
            ]

        if transform_args.random_crop is not None:
            transform += [
                RandomCrop(
                    height=no_auto(transform_args.random_crop.height),
                    width=no_auto(transform_args.random_crop.width),
                    pad_if_needed=transform_args.random_crop.pad_if_needed,
                    pad_position=transform_args.random_crop.pad_position,
                    fill=transform_args.random_crop.fill,
                    fill_mask=transform_args.ignore_index,
                    p=transform_args.random_crop.prob,
                )
            ]

        # Optionally apply random horizontal flip.
        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                transform += [
                    HorizontalFlip(p=transform_args.random_flip.horizontal_prob)
                ]
            if transform_args.random_flip.vertical_prob > 0.0:
                transform += [VerticalFlip(p=transform_args.random_flip.vertical_prob)]

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
                mean=no_auto(transform_args.normalize).mean,
                std=no_auto(transform_args.normalize).std,
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
