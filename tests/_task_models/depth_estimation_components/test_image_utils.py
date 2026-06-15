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
from torch import Tensor

from lightly_train._task_models.depth_estimation_components import image_utils


def test__resize_bound__upper_bound_resizes_longest_side() -> None:
    img = torch.full((3, 8, 4), 100.0)
    out = image_utils._resize_bound(img, target_size=4, method="upper_bound_resize")
    assert out.shape == (3, 4, 2)
    assert torch.equal(out, torch.full((3, 4, 2), 100.0))


def test__resize_bound__lower_bound_resizes_shortest_side() -> None:
    img = torch.full((3, 8, 4), 100.0)
    out = image_utils._resize_bound(img, target_size=2, method="lower_bound_resize")
    assert out.shape == (3, 4, 2)
    assert torch.equal(out, torch.full((3, 4, 2), 100.0))


def test__resize__area_matches_cv2_bit_exactly() -> None:
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (3, 30, 48), generator=generator, dtype=torch.uint8)
    out = image_utils._resize(
        img.float(), new_h=20, new_w=32, method="area", round_to_grid=True
    )
    expected = _cv2_resize(img, new_h=20, new_w=32, interpolation=cv2.INTER_AREA)
    assert torch.equal(out, expected.round().clamp(min=0.0, max=255.0))


def test__resize__cubic_matches_cv2_up_to_rounding_ties() -> None:
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (3, 32, 48), generator=generator, dtype=torch.uint8)
    out = image_utils._resize(
        img.float(), new_h=48, new_w=72, method="cubic", round_to_grid=True
    )
    expected = _cv2_resize(img, new_h=48, new_w=72, interpolation=cv2.INTER_CUBIC)
    expected_rounded = expected.round().clamp(min=0.0, max=255.0)
    # Cubic weights at half-pixel offsets are dyadic, so a pre-rounding value can
    # land exactly on a .5 tie where the float accumulation order may flip the
    # rounding by one. Non-tie pixels must match bit-exactly; tie pixels may
    # differ by at most one and must be rare.
    tie = (expected - expected.floor() - 0.5).abs() < 1e-3
    mismatch = out != expected_rounded
    assert not (mismatch & ~tie).any()
    assert (out - expected_rounded).abs().max() <= 1.0
    assert mismatch.float().mean() < 0.01


def test__resize__unsupported_method_raises() -> None:
    img = torch.zeros(3, 4, 4)

    with pytest.raises(ValueError, match="Unsupported resize method 'nearest'"):
        image_utils._resize(img, new_h=2, new_w=2, method="nearest", round_to_grid=True)


def test__resize_to_patch_multiple__rounds_each_side_to_nearest_multiple() -> None:
    img = torch.full((3, 45, 56), 100.0)

    out = image_utils._resize_to_patch_multiple(img, patch=14)

    # 45 rounds down to 42; 56 is already a multiple of 14 and stays unchanged.
    assert out.shape == (3, 42, 56)
    assert torch.equal(out, torch.full((3, 42, 56), 100.0))


def test__resize_to_patch_multiple__halfway_rounds_up() -> None:
    img = torch.full((3, 21, 28), 100.0)

    out = image_utils._resize_to_patch_multiple(img, patch=14)

    # 21 is equally far from 14 and 28; the tie rounds up.
    assert out.shape == (3, 28, 28)
    assert torch.equal(out, torch.full((3, 28, 28), 100.0))


def test__dav2_target_size__lower_bound_rounds_to_multiple_of_14() -> None:
    # The shorter side (480) scales to input_size (518); the longer side scales by the
    # same factor (690.67) and rounds to the nearest multiple of 14 (686).
    new_h, new_w = image_utils._dav2_target_size(h=480, w=640, input_size=518, patch=14)
    assert (new_h, new_w) == (518, 686)


def test__dav2_target_size__square_maps_to_input_size() -> None:
    new_h, new_w = image_utils._dav2_target_size(h=500, w=500, input_size=518, patch=14)
    assert (new_h, new_w) == (518, 518)


def test__dav2_target_size__ceils_when_below_min_val() -> None:
    # With an input_size that is not a multiple of patch, rounding the shorter side down
    # would fall below input_size, so it must ceil up to the next multiple (532).
    new_h, new_w = image_utils._dav2_target_size(h=500, w=500, input_size=520, patch=14)
    assert (new_h, new_w) == (532, 532)


def test__resize__cubic_no_round_matches_cv2_float64() -> None:
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (3, 30, 48), generator=generator, dtype=torch.uint8)
    img64 = img.to(torch.float64)

    out = image_utils._resize(
        img64, new_h=42, new_w=70, method="cubic", round_to_grid=False
    )

    expected = _cv2_resize_f64(img64, new_h=42, new_w=70, interpolation=cv2.INTER_CUBIC)
    # cv2 computes the cubic source coordinate and coefficients in float32 internally
    # (even for float64 data), so a dependency-free float64 reproduction matches cv2 to
    # ~1e-2 in [0, 255] space rather than bit-exactly. A wrong kernel/coordinate would
    # differ by whole pixel values, so this still strongly pins the implementation.
    assert out.dtype == torch.float64
    assert (out - expected).abs().max() < 0.05
    assert (out - expected).abs().mean() < 5e-3


def test_process_image_dav2__matches_official_cubic_pipeline() -> None:
    generator = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (3, 360, 540), generator=generator, dtype=torch.uint8)

    out = image_utils.process_image_dav2(img, input_size=518, patch=14)

    new_h, new_w = image_utils._dav2_target_size(h=360, w=540, input_size=518, patch=14)
    assert out.shape == (3, new_h, new_w)
    # The official pipeline resizes the float image with cv2.INTER_CUBIC without
    # rounding back to the integer grid; comparing in [0, 255] space is equivalent to
    # the official /255-then-resize because the resize is linear. See the note in
    # test__resize__cubic_no_round_matches_cv2_float64 on the cv2 float32 internals.
    expected = _cv2_resize_f64(
        img.to(torch.float64), new_h=new_h, new_w=new_w, interpolation=cv2.INTER_CUBIC
    )
    assert (out - expected).abs().max() < 0.05
    assert (out - expected).abs().mean() < 5e-3


def _cv2_resize(img: Tensor, *, new_h: int, new_w: int, interpolation: int) -> Tensor:
    """Resizes a ``(C, H, W)`` tensor with cv2 in float32, as the official DA3
    pipeline does, and returns the unrounded float ``(C, H, W)`` result."""
    img_np = img.permute(1, 2, 0).numpy().astype(np.float32)
    resized = cv2.resize(src=img_np, dsize=(new_w, new_h), interpolation=interpolation)
    return torch.from_numpy(resized).permute(2, 0, 1)


def _cv2_resize_f64(
    img: Tensor, *, new_h: int, new_w: int, interpolation: int
) -> Tensor:
    """Resizes a ``(C, H, W)`` tensor with cv2 in float64, as the official DA2
    pipeline does, and returns the unrounded float ``(C, H, W)`` result."""
    img_np = img.permute(1, 2, 0).numpy().astype(np.float64)
    resized = cv2.resize(src=img_np, dsize=(new_w, new_h), interpolation=interpolation)
    return torch.from_numpy(resized).permute(2, 0, 1)
