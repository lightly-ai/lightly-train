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

from lightly_train._task_models.depth_estimation_components import image_utils


class TestProcessImage:
    def test_process_image__upper_bound_resize(self) -> None:
        # 64x100 image, longest side (100) scaled to 42, aspect preserved (64 -> ~27),
        # then both sides rounded to the nearest multiple of 14.
        img = torch.randint(0, 256, (3, 64, 100), dtype=torch.uint8)

        out = image_utils.process_image(
            img, process_res=42, process_res_method="upper_bound_resize"
        )

        assert out.dtype == torch.float32
        h, w = out.shape[-2:]
        assert h % image_utils.PATCH_SIZE == 0
        assert w % image_utils.PATCH_SIZE == 0
        # The longest side reaches the target multiple; height stays smaller.
        assert w == 42
        assert h < w

    def test_process_image__lower_bound_resize(self) -> None:
        # 64x100 image, shortest side (64) scaled to 42, aspect preserved, both sides
        # rounded to a multiple of 14.
        img = torch.randint(0, 256, (3, 64, 100), dtype=torch.uint8)

        out = image_utils.process_image(
            img, process_res=42, process_res_method="lower_bound_resize"
        )

        assert out.dtype == torch.float32
        h, w = out.shape[-2:]
        assert h % image_utils.PATCH_SIZE == 0
        assert w % image_utils.PATCH_SIZE == 0
        # The shortest side reaches the target multiple; width stays larger.
        assert h == 42
        assert w > h

    def test_process_image__square_resize(self) -> None:
        # Any input aspect ratio maps to an exact (process_res, process_res) square.
        img = torch.randint(0, 256, (3, 64, 100), dtype=torch.uint8)

        out = image_utils.process_image(
            img, process_res=56, process_res_method="square_resize"
        )

        assert out.shape == (3, 56, 56)
        assert out.dtype == torch.float32

    def test_process_image__unsupported_method_raises(self) -> None:
        img = torch.zeros(3, 32, 32, dtype=torch.uint8)

        with pytest.raises(ValueError, match="Unsupported process_res_method 'nope'"):
            image_utils.process_image(img, process_res=28, process_res_method="nope")

    def test_process_image__grayscale_expanded_to_rgb(self) -> None:
        img = torch.full((1, 28, 28), 128, dtype=torch.uint8)

        out = image_utils.process_image(
            img, process_res=28, process_res_method="square_resize"
        )

        assert out.shape == (3, 28, 28)

    def test_process_image__alpha_channel_dropped(self) -> None:
        img = torch.randint(0, 256, (4, 28, 28), dtype=torch.uint8)

        out = image_utils.process_image(
            img, process_res=28, process_res_method="square_resize"
        )

        assert out.shape == (3, 28, 28)

    def test_process_image__unsupported_channels_raises(self) -> None:
        img = torch.zeros(2, 28, 28, dtype=torch.uint8)

        with pytest.raises(ValueError, match="1, 3, or 4 channels"):
            image_utils.process_image(
                img, process_res=28, process_res_method="square_resize"
            )


def test__resize_bound__upper_bound_resizes_longest_side() -> None:
    img = torch.full((3, 8, 4), 100.0)
    out = image_utils._resize_bound(img, target_size=4, method="upper_bound_resize")
    assert out.shape == (3, 4, 2)


def test__resize_bound__lower_bound_resizes_shortest_side() -> None:
    img = torch.full((3, 8, 4), 100.0)
    out = image_utils._resize_bound(img, target_size=2, method="lower_bound_resize")
    assert out.shape == (3, 4, 2)


def test__resize_to_patch_multiple__rounds_each_side_to_nearest_multiple() -> None:
    img = torch.full((3, 45, 56), 100.0)

    out = image_utils._resize_to_patch_multiple(img, patch=14)

    # 45 rounds down to 42; 56 is already a multiple of 14 and stays unchanged.
    assert out.shape == (3, 42, 56)


def test__resize_to_patch_multiple__halfway_rounds_up() -> None:
    img = torch.full((3, 21, 28), 100.0)

    out = image_utils._resize_to_patch_multiple(img, patch=14)

    # 21 is equally far from 14 and 28; the tie rounds up.
    assert out.shape == (3, 28, 28)
