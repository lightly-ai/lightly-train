#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
from pydantic import ValidationError

from lightly_train._transforms.random_resized_crop import get_random_resized_crop
from lightly_train._transforms.transform import RandomResizedCropArgs


def test_get_random_resized_crop() -> None:
    args = RandomResizedCropArgs(min_scale=0.2, max_scale=1.0)
    crop = get_random_resized_crop(size=(64, 64), args=args)
    assert crop.scale == (0.2, 1.0)
    assert crop.ratio == (3 / 4, 4 / 3)


def test_get_random_resized_crop__ratio() -> None:
    args = RandomResizedCropArgs(
        min_scale=0.2, max_scale=1.0, min_ratio=0.5, max_ratio=2.0
    )
    crop = get_random_resized_crop(size=(64, 64), args=args)
    assert crop.ratio == (0.5, 2.0)


@pytest.mark.parametrize("ratio", [0.0, -1.0])
def test_random_resized_crop_args__non_positive_ratio(ratio: float) -> None:
    with pytest.raises(ValidationError):
        RandomResizedCropArgs(min_ratio=ratio)
    with pytest.raises(ValidationError):
        RandomResizedCropArgs(max_ratio=ratio)
