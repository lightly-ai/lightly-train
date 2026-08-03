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

import numpy as np
from albumentations import (
    BasicTransform,
    ColorJitter,
    HorizontalFlip,
    OneOf,
    RandomCrop,
    RandomRotate90,
    Resize,
    Rotate,
    SmallestMaxSize,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.transform import NormalizeArgs

logger = logging.getLogger(__name__)


def _resolve_incompatible_eomt(transform_args: Any) -> None:
    transform_args.normalize = no_auto(transform_args.normalize)
    assert isinstance(transform_args.normalize, NormalizeArgs)
    num_channels = no_auto(transform_args.num_channels)
    assert isinstance(num_channels, int)

    # Adjust normalization mean and std to match num_channels.
    if len(transform_args.normalize.mean) != num_channels:
        logger.debug(
            "Adjusting mean of normalize transform to match num_channels. "
            f"num_channels is {num_channels} but "
            f"normalize.mean has length {len(transform_args.normalize.mean)}."
        )
        # Repeat the values until they match num_channels.
        transform_args.normalize.mean = tuple(
            transform_args.normalize.mean[i % len(transform_args.normalize.mean)]
            for i in range(num_channels)
        )
    if len(transform_args.normalize.std) != num_channels:
        logger.debug(
            "Adjusting std of normalize transform to match num_channels. "
            f"num_channels is {num_channels} but "
            f"normalize.std has length {len(transform_args.normalize.std)}."
        )
        # Repeat the values until they match num_channels.
        transform_args.normalize.std = tuple(
            transform_args.normalize.std[i % len(transform_args.normalize.std)]
            for i in range(num_channels)
        )

    # Disable color jitter if necessary.
    if transform_args.color_jitter is not None and num_channels != 3:
        logger.debug(
            "Disabling color jitter transform as it only supports 3-channel "
            f"images but num_channels is {num_channels}."
        )
        transform_args.color_jitter = None


def _build_transforms_eomt(
    transform_args: Any,
    *,
    random_crop_fill_mask: int | None = None,
) -> list[BasicTransform]:
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
        assert isinstance(transform_args.image_size, tuple)
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
        random_crop_kwargs: dict[str, Any] = {}
        if random_crop_fill_mask is not None:
            random_crop_kwargs["fill_mask"] = random_crop_fill_mask
        transform += [
            RandomCrop(
                height=no_auto(transform_args.random_crop.height),
                width=no_auto(transform_args.random_crop.width),
                pad_if_needed=transform_args.random_crop.pad_if_needed,
                pad_position=transform_args.random_crop.pad_position,
                fill=transform_args.random_crop.fill,
                p=transform_args.random_crop.prob,
                **random_crop_kwargs,
            )
        ]

    # Optionally apply random horizontal flip.
    if transform_args.random_flip is not None:
        if transform_args.random_flip.horizontal_prob > 0.0:
            transform += [HorizontalFlip(p=transform_args.random_flip.horizontal_prob)]
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

    return transform
