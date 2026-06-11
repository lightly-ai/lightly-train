#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from lightly_train._task_models.depth_estimation_components import image_utils


def test__resize_bound__upper_bound_resizes_longest_side() -> None:
    img = torch.full((3, 8, 4), 100.0)

    out = image_utils._resize_bound(img, target_size=4, method="upper_bound_resize")

    # The longest side (8) is bounded to 4; the aspect ratio is preserved. A constant
    # image must stay constant because the resize weights sum to one.
    assert out.shape == (3, 4, 2)
    assert torch.equal(out, torch.full((3, 4, 2), 100.0))


def test__resize_bound__lower_bound_resizes_shortest_side() -> None:
    img = torch.full((3, 8, 4), 100.0)

    out = image_utils._resize_bound(img, target_size=2, method="lower_bound_resize")

    # The shortest side (4) is bounded to 2; the aspect ratio is preserved.
    assert out.shape == (3, 4, 2)
    assert torch.equal(out, torch.full((3, 4, 2), 100.0))


def test__resize_to__area_shrink_averages_blocks() -> None:
    # A 2x downscale with INTER_AREA is the mean over each 2x2 block:
    # (10 + 20 + 50 + 60) / 4 = 35 and (30 + 40 + 70 + 80) / 4 = 55.
    img = torch.tensor([[[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]])

    out = image_utils._resize_to(img, new_h=1, new_w=2, method="area")

    assert torch.equal(out, torch.tensor([[[35.0, 55.0]]]))


def test__resize_to__cubic_enlarge_clamps_overshoot() -> None:
    # A 2x cubic upscale of [10, 250]: the Keys kernel (a = -0.75) with
    # replicate-border taps gives per-output weights [1.10546875, -0.10546875],
    # [0.7734375, 0.2265625], and their mirrors, so the outputs are -15.3125,
    # 64.375, 195.625, and 275.3125, rounded and clamped to [0, 255].
    img = torch.tensor([[[10.0, 250.0]]])

    out = image_utils._resize_to(img, new_h=1, new_w=4, method="cubic")

    assert torch.equal(out, torch.tensor([[[0.0, 64.0, 196.0, 255.0]]]))


def test__resize_to__area_matches_cv2_bit_exactly() -> None:
    # A 1.5x downscale: every output pixel is an average with weights in thirds, so
    # no exact value can land on a .5 rounding tie and the result must match cv2's
    # INTER_AREA (computed in float, then rounded back to the integer grid)
    # bit-exactly.
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (3, 30, 48), generator=generator, dtype=torch.uint8)
    img_np = img.permute(1, 2, 0).numpy().astype(np.float32)

    out = image_utils._resize_to(img.float(), new_h=20, new_w=32, method="area")

    expected = cv2.resize(img_np, (32, 20), interpolation=cv2.INTER_AREA)
    expected_rounded = (
        torch.from_numpy(expected).permute(2, 0, 1).round().clamp(min=0.0, max=255.0)
    )
    assert torch.equal(out, expected_rounded)


def test__resize_to__cubic_matches_cv2_up_to_rounding_ties() -> None:
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (3, 32, 48), generator=generator, dtype=torch.uint8)
    img_np = img.permute(1, 2, 0).numpy().astype(np.float32)

    out = image_utils._resize_to(img.float(), new_h=48, new_w=72, method="cubic")

    expected = torch.from_numpy(
        cv2.resize(img_np, (72, 48), interpolation=cv2.INTER_CUBIC)
    ).permute(2, 0, 1)
    expected_rounded = expected.round().clamp(min=0.0, max=255.0)
    # Cubic weights are dyadic, so cv2's pre-rounding value can land exactly on a
    # .5 tie; there the float accumulation order may flip the rounding by one.
    # Every other pixel must match bit-exactly, and ties must be rare.
    tie = (expected - expected.floor() - 0.5).abs() < 1e-3
    mismatch = out != expected_rounded
    assert not (mismatch & ~tie).any()
    assert (out - expected_rounded).abs().max() <= 1.0
    assert mismatch.float().mean() < 0.01


def test__resize_to__unsupported_method_raises() -> None:
    img = torch.zeros(3, 4, 4)

    with pytest.raises(ValueError, match="Unsupported resize method 'nearest'"):
        image_utils._resize_to(img, new_h=2, new_w=2, method="nearest")


def test__make_divisible_by_resize__rounds_each_side_to_nearest_multiple() -> None:
    img = torch.full((3, 45, 56), 100.0)

    out = image_utils._make_divisible_by_resize(img, patch=14)

    # 45 rounds down to 42; 56 is already a multiple of 14.
    assert out.shape == (3, 42, 56)
    assert torch.equal(out, torch.full((3, 42, 56), 100.0))


def test__make_divisible_by_resize__halfway_rounds_up() -> None:
    img = torch.full((3, 21, 28), 100.0)

    out = image_utils._make_divisible_by_resize(img, patch=14)

    # 21 is equally far from 14 and 28; the tie rounds up.
    assert out.shape == (3, 28, 28)
    assert torch.equal(out, torch.full((3, 28, 28), 100.0))
