#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import itertools
from typing import Tuple, Union

import numpy as np
import pytest
import torch

from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomResizedCropArgs,
    RandomRotationArgs,
    SolarizeArgs,
)
from lightly_train._transforms.view_transform import ViewTransform, ViewTransformArgs
from lightly_train.types import TransformInput


def _get_channel_drop_args() -> ChannelDropArgs:
    return ChannelDropArgs(
        num_channels_keep=3,
        weight_drop=(1.0, 1.0, 0.0, 0.0),
    )


def _get_random_resized_crop_args() -> RandomResizedCropArgs:
    return RandomResizedCropArgs(
        size=(64, 64),
        scale=RandomResizeArgs(min_scale=0.2, max_scale=1.0),
    )


def _get_random_flip_args() -> RandomFlipArgs:
    return RandomFlipArgs(horizontal_prob=0.5, vertical_prob=0.5)


def _get_random_rotation_args() -> RandomRotationArgs:
    return RandomRotationArgs(prob=0.5, degrees=10)


def _get_color_jitter_args() -> ColorJitterArgs:
    return ColorJitterArgs(
        prob=0.8,
        strength=1.0,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1,
    )


def _get_random_gray_scale() -> float:
    return 0.2


def _get_gaussian_blur_args() -> GaussianBlurArgs:
    return GaussianBlurArgs(
        prob=0.5,
        sigmas=(0.1, 2),
        blur_limit=(3, 7),
    )


def _get_solarize_args() -> SolarizeArgs:
    return SolarizeArgs(prob=0.5, threshold=0.5)


def _get_normalize_args() -> NormalizeArgs:
    return NormalizeArgs(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


PossibleArgsTuple = Tuple[
    Union[ChannelDropArgs, None],
    RandomResizedCropArgs,
    Union[RandomFlipArgs, None],
    Union[RandomRotationArgs, None],
    Union[ColorJitterArgs, None],
    Union[float, None],
    Union[GaussianBlurArgs, None],
    Union[SolarizeArgs, None],
    NormalizeArgs,
]


def _get_possible_view_transform_args_combinations() -> list[PossibleArgsTuple]:
    channel_drop = [_get_channel_drop_args(), None]
    random_resized_crop = [_get_random_resized_crop_args()] * 2
    random_flip = [_get_random_flip_args(), None]
    random_rotation = [_get_random_rotation_args(), None]
    color_jitter = [_get_color_jitter_args(), None]
    random_gray_scale = [_get_random_gray_scale(), None]
    gaussian_blur = [_get_gaussian_blur_args(), None]
    solarize = [_get_solarize_args(), None]
    normalize = [_get_normalize_args()] * 2
    return list(
        itertools.product(
            channel_drop,
            random_resized_crop,
            random_flip,
            random_rotation,
            color_jitter,
            random_gray_scale,
            gaussian_blur,
            solarize,
            normalize,
        )
    )


possible_tuples = _get_possible_view_transform_args_combinations()


class TestViewTransform:
    @pytest.mark.parametrize(
        "channel_drop, random_resized_crop, random_flip, random_rotation, color_jitter, random_gray_scale, gaussian_blur, solarize, normalize",
        possible_tuples,
    )
    def test_view_transform_all_args_combinations(
        self,
        channel_drop: ChannelDropArgs | None,
        random_resized_crop: RandomResizedCropArgs,
        random_flip: RandomFlipArgs | None,
        random_rotation: RandomRotationArgs | None,
        color_jitter: ColorJitterArgs | None,
        random_gray_scale: float | None,
        gaussian_blur: GaussianBlurArgs | None,
        solarize: SolarizeArgs | None,
        normalize: NormalizeArgs,
    ) -> None:
        view_transform = ViewTransform(
            ViewTransformArgs(
                channel_drop=channel_drop,
                random_resized_crop=random_resized_crop,
                random_flip=random_flip,
                random_rotation=random_rotation,
                color_jitter=color_jitter,
                random_gray_scale=random_gray_scale,
                gaussian_blur=gaussian_blur,
                solarize=solarize,
                normalize=normalize,
            )
        )
        num_channels = 3 if channel_drop is None else 4
        tr_input: TransformInput = {
            "image": np.random.rand(224, 224, num_channels).astype(np.float32),
        }
        tr_output = view_transform(tr_input)
        assert isinstance(tr_output, dict)
        img = tr_output["image"]
        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32
