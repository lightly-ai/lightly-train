#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors

from lightly_train._transforms.random_rotate_90 import RandomRotate90


@pytest.fixture
def dummy_tv_image() -> tv_tensors.Image:
    return tv_tensors.Image(torch.rand(3, 64, 64))


class TestRandomRotate90:
    def test_forward_image_shape(self, dummy_tv_image: tv_tensors.Image) -> None:
        transform = RandomRotate90(p=1.0)
        result = transform(dummy_tv_image)
        assert result.shape == dummy_tv_image.shape

    def test_forward_multiple_iterations(
        self, dummy_tv_image: tv_tensors.Image
    ) -> None:
        transform = RandomRotate90(p=0.5)
        for _ in range(10):
            result = transform(dummy_tv_image)
            assert result.shape == dummy_tv_image.shape

    def test_forward_preserves_dtype(self, dummy_tv_image: tv_tensors.Image) -> None:
        transform = RandomRotate90(p=1.0)
        result = transform(dummy_tv_image)
        assert result.dtype == dummy_tv_image.dtype

    def test_forward_with_p_zero_returns_same(
        self, dummy_tv_image: tv_tensors.Image
    ) -> None:
        transform = RandomRotate90(p=0.0)
        result = transform(dummy_tv_image)
        assert torch.equal(result, dummy_tv_image)

    def test_forward_with_multiple_images(self) -> None:
        transform = RandomRotate90(p=0.5)
        images = [
            tv_tensors.Image(torch.rand(3, 32, 32)),
            tv_tensors.Image(torch.rand(3, 64, 64)),
        ]
        results = transform(*images)
        assert len(results) == len(images)

    def test_forward_gray_image(self) -> None:
        transform = RandomRotate90(p=1.0)
        gray_image = tv_tensors.Image(torch.rand(1, 64, 64))
        result = transform(gray_image)
        assert result.shape == gray_image.shape
