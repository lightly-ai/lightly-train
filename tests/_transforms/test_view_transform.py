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
from lightning_utilities.core.imports import RequirementCache

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

ALBUMENTATIONS_VERSION_2XX = RequirementCache("albumentations>=2.0.0")
RECORD_GEOMETRY_SKIP = pytest.mark.skipif(
    not ALBUMENTATIONS_VERSION_2XX,
    reason="record_geometry requires albumentations>=2.0.0",
)


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
    @RECORD_GEOMETRY_SKIP
    def test_record_geometry_attaches_geometry_tensor(self) -> None:
        # record_geometry=True records the crop box and flips applied to a view
        # as an 8-element tensor [x0, y0, x1, y1, image_w, image_h, hflip, vflip].
        view_transform = ViewTransform(
            ViewTransformArgs(
                channel_drop=None,
                random_resized_crop=_get_random_resized_crop_args(),
                random_flip=_get_random_flip_args(),
                random_rotation=None,
                color_jitter=_get_color_jitter_args(),
                random_gray_scale=_get_random_gray_scale(),
                gaussian_blur=_get_gaussian_blur_args(),
                solarize=None,
                normalize=_get_normalize_args(),
            ),
            record_geometry=True,
        )
        image_w, image_h = 224, 224
        tr_input: TransformInput = {
            "image": np.random.rand(image_h, image_w, 3).astype(np.float32),
        }
        tr_output = view_transform(tr_input)
        assert "geometry" in tr_output
        geometry = tr_output["geometry"]
        assert geometry.shape == (8,)
        assert geometry.dtype == torch.float32
        x0, y0, x1, y1, w, h, hflip, vflip = geometry.tolist()
        # Crop box lies inside the original image; flip flags are 0/1.
        assert 0 <= x0 < x1 <= image_w
        assert 0 <= y0 < y1 <= image_h
        assert w == image_w and h == image_h
        assert hflip in (0.0, 1.0) and vflip in (0.0, 1.0)

    @RECORD_GEOMETRY_SKIP
    def test_record_geometry_reusable_across_views(self) -> None:
        # A single ViewTransform is reused sequentially across views (as the
        # DINO multi-crop transform does). Each call must record its own geometry
        # independently, with no applied_transforms key leaking into the output.
        view_transform = ViewTransform(
            ViewTransformArgs(
                channel_drop=None,
                random_resized_crop=_get_random_resized_crop_args(),
                random_flip=_get_random_flip_args(),
                random_rotation=None,
                color_jitter=None,
                random_gray_scale=None,
                gaussian_blur=None,
                solarize=None,
                normalize=_get_normalize_args(),
            ),
            record_geometry=True,
        )
        tr_input: TransformInput = {
            "image": np.random.rand(128, 128, 3).astype(np.float32),
        }
        out0 = view_transform(tr_input)
        out1 = view_transform(tr_input)
        for out in (out0, out1):
            assert "geometry" in out
            assert "applied_transforms" not in out
            assert out["geometry"].shape == (8,)

    @RECORD_GEOMETRY_SKIP
    def test_record_geometry_disallowed_with_rotation(self) -> None:
        with pytest.raises(ValueError, match="random_rotation"):
            ViewTransform(
                ViewTransformArgs(
                    channel_drop=None,
                    random_resized_crop=_get_random_resized_crop_args(),
                    random_flip=_get_random_flip_args(),
                    random_rotation=_get_random_rotation_args(),
                    color_jitter=None,
                    random_gray_scale=None,
                    gaussian_blur=None,
                    solarize=None,
                    normalize=_get_normalize_args(),
                ),
                record_geometry=True,
            )

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
