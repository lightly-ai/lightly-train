#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any

import cv2
from albumentations import (
    BasicTransform,
    CenterCrop,
    ColorJitter,
    Compose,
    HorizontalFlip,
    RandomResizedCrop,
    RandomRotate90,
    Resize,
    Rotate,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2
from torch import Tensor
from typing_extensions import Literal

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
    RandomFlipArgs,
    RandomResizeArgs,
    RandomRotate90Args,
    RandomRotationArgs,
)
from lightly_train.types import (
    ImageSizeTuple,
    NDArrayImage,
)

logger = logging.getLogger(__name__)


class ImageClassificationTransformInput(TaskTransformInput):
    image: NDArrayImage


class ImageClassificationTransformOutput(TaskTransformOutput):
    image: Tensor


class ImageClassificationTransformArgs(TaskTransformArgs):
    image_size: ImageSizeTuple | Literal["auto"]
    channel_drop: ChannelDropArgs | None
    num_channels: int | Literal["auto"]
    normalize: NormalizeArgs | Literal["auto"]
    random_crop: RandomResizeArgs | None
    resize_scale: float | None
    random_flip: RandomFlipArgs | None
    random_rotate_90: RandomRotate90Args | None
    random_rotate: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None

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


class ImageClassificationTransform(TaskTransform):
    transform_args_cls: type[ImageClassificationTransformArgs] = (
        ImageClassificationTransformArgs
    )

    def __init__(
        self,
        transform_args: ImageClassificationTransformArgs,
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

        if transform_args.random_crop is not None:
            transform += [
                RandomResizedCrop(
                    size=no_auto(transform_args.image_size),
                    scale=(
                        transform_args.random_crop.min_scale,
                        transform_args.random_crop.max_scale,
                    ),
                    # Use inter area based on https://github.com/lightly-ai/lightly-train-old/pull/284
                    interpolation=cv2.INTER_AREA,
                )
            ]
        else:
            # Default ImageNet classification resize for validation:
            # 1. Resize the shorter side to 256 (= 224 * 1.14) with aspect ratio kept.
            # 2. Center crop to 224x224.
            height, width = no_auto(transform_args.image_size)
            resize_scale = no_auto(transform_args.resize_scale)
            resize_height = height
            resize_width = width
            if resize_scale is not None:
                resize_height = int(height * resize_scale)
                resize_width = int(width * resize_scale)
            transform += [
                Resize(
                    height=resize_height,
                    width=resize_width,
                    interpolation=cv2.INTER_AREA,
                )
            ]
            if resize_scale is not None:
                transform += [
                    CenterCrop(
                        height=height,
                        width=width,
                    )
                ]

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                transform += [
                    HorizontalFlip(p=transform_args.random_flip.horizontal_prob)
                ]
            if transform_args.random_flip.vertical_prob > 0.0:
                transform += [VerticalFlip(p=transform_args.random_flip.vertical_prob)]

        if transform_args.random_rotate_90 is not None:
            transform += [RandomRotate90(p=transform_args.random_rotate_90.prob)]

        if transform_args.random_rotate is not None:
            transform += [
                Rotate(
                    limit=transform_args.random_rotate.degrees,
                    interpolation=transform_args.random_rotate.interpolation,
                    p=transform_args.random_rotate.prob,
                )
            ]

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

        transform += [
            Normalize(
                mean=no_auto(transform_args.normalize).mean,
                std=no_auto(transform_args.normalize).std,
            )
        ]

        transform += [ToTensorV2()]

        self.transform = Compose(transform)

    def __call__(
        self, input: ImageClassificationTransformInput
    ) -> ImageClassificationTransformOutput:
        transformed = self.transform(image=input["image"])
        return {
            "image": transformed["image"],
        }
