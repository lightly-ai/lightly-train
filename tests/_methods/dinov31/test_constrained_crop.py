#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from lightly_train._methods.dinov31.constrained_crop import (
    render_clean_global,
    sample_high_overlap_box,
)
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import TransformInput


class TestSampleHighOverlapBox:
    @pytest.mark.parametrize("min_iou", [0.5, 0.7, 0.9, 0.95])
    @pytest.mark.parametrize(
        "parent_box",
        [(0, 0, 200, 200), (10, 30, 410, 230), (5, 5, 105, 405), (0, 0, 50, 400)],
    )
    def test_child_inside_parent_small_and_contained(
        self, min_iou: float, parent_box: tuple[int, int, int, int]
    ) -> None:
        # The local is a genuine small crop fully inside the parent (containment
        # is the PaKA paper's overlap criterion; see sample_high_overlap_box).
        rng = random.Random(0)
        px0, py0, px1, py1 = parent_box
        parent_area = (px1 - px0) * (py1 - py0)
        for _ in range(500):
            x0, y0, x1, y1 = sample_high_overlap_box(parent_box, min_iou, rng)
            assert px0 <= x0 < x1 <= px1
            assert py0 <= y0 < y1 <= py1
            local_area = (x1 - x0) * (y1 - y0)
            assert local_area / parent_area <= 0.5

    def test_degenerate_parent_returns_parent(self) -> None:
        rng = random.Random(0)
        assert sample_high_overlap_box((5, 5, 6, 100), 0.9, rng) == (5, 5, 6, 100)
        assert sample_high_overlap_box((5, 5, 100, 6), 0.9, rng) == (5, 5, 100, 6)

    def test_deterministic_given_seed(self) -> None:
        b1 = sample_high_overlap_box((0, 0, 300, 300), 0.9, random.Random(42))
        b2 = sample_high_overlap_box((0, 0, 300, 300), 0.9, random.Random(42))
        assert b1 == b2


class TestRenderCleanGlobal:
    def _input(self) -> TransformInput:
        rng = np.random.default_rng(0)
        return {"image": (rng.random((480, 640, 3)) * 255).astype(np.uint8)}

    def test_output_shape_and_geometry_preserved(self) -> None:
        geometry = torch.tensor(
            [12.0, 30.0, 412.0, 330.0, 640.0, 480.0, 0.0, 0.0], dtype=torch.float32
        )
        out = render_clean_global(
            input=self._input(),
            geometry=geometry,
            size=(224, 224),
            normalize=NormalizeArgs(),
        )
        assert out["image"].shape == (3, 224, 224)
        # Geometry is re-used verbatim so it matches the source global crop.
        assert torch.equal(out["geometry"], geometry)

    def test_hflip_flips_the_image(self) -> None:
        # The recorded hflip flag must actually flip the rendered pixels.
        geom_base = torch.tensor(
            [0.0, 0.0, 320.0, 240.0, 640.0, 480.0, 0.0, 0.0], dtype=torch.float32
        )
        geom_hflip = geom_base.clone()
        geom_hflip[6] = 1.0
        out = render_clean_global(
            input=self._input(),
            geometry=geom_base,
            size=(98, 98),
            normalize=NormalizeArgs(),
        )
        out_hflip = render_clean_global(
            input=self._input(),
            geometry=geom_hflip,
            size=(98, 98),
            normalize=NormalizeArgs(),
        )
        assert out["image"].shape == (3, 98, 98)
        assert not torch.equal(out["image"], out_hflip["image"])
