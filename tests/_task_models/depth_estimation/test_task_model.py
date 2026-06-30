#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import pytest
import torch
from PIL import Image

from lightly_train._task_models.depth_estimation import task_model
from lightly_train._task_models.depth_estimation.task_model import (
    DepthAnythingDepthEstimation,
)

# Tiny per-variant model args so inference runs at a small resolution and tests stay
# fast. DAv3 variants carry `use_sky_head` inside `model_args` exactly as the exported
# checkpoints do; DAv2 metric variants carry `max_depth`.
_TINY_BASE: dict[str, Any] = {
    "out_layers": (0, 1, 2, 3),
    "image_size": 56,
    "patch_size": 14,
    "features": 16,
    "out_channels": (8, 16, 32, 32),
}


def _tiny_model_args(model_name: str) -> dict[str, Any]:
    args = dict(_TINY_BASE)
    if "metric" in model_name and "dav2" in model_name:
        args["max_depth"] = 20.0
    if "dav3" in model_name:
        args["use_sky_head"] = True
    return args


def _build(model_name: str, **overrides: Any) -> DepthAnythingDepthEstimation:
    model_args = {**_tiny_model_args(model_name), **overrides}
    model = DepthAnythingDepthEstimation(
        model_name=model_name,
        model_args=model_args,
        load_weights=False,
    )
    # Production fixes the image size per model; override it here so inference runs
    # at a tiny resolution and the tests stay fast.
    model.image_size = 56
    return model


# Model names that do not take intrinsics (relative + DAv2 metric).
_NON_FOCAL_NAMES = [
    "dinov2/dav2-relative-large",
    "dinov2/dav2-metric-large-hypersim",
    "dinov2/dav3-relative-large",
]
# Model names with a sky head (DAv3).
_SKY_NAMES = ["dinov2/dav3-relative-large", "dinov2/dav3-metric-large"]
_NO_SKY_NAMES = ["dinov2/dav2-relative-large", "dinov2/dav2-metric-large-hypersim"]


class TestDepthAnythingDepthEstimation:
    @pytest.mark.parametrize("model_name", _NON_FOCAL_NAMES)
    def test_predict__returns_original_resolution(self, model_name: str) -> None:
        model = _build(model_name)
        image = Image.new("RGB", (80, 64), color=(32, 64, 128))

        depth = model.predict(image)

        # Inference runs at the processing resolution and the depth map is resized back
        # to the original input resolution.
        assert depth.shape == (64, 80)
        assert depth.dtype == next(model.parameters()).dtype
        assert torch.isfinite(depth).all()

    def test_predict__returns_original_resolution_focal(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        image = Image.new("RGB", (80, 64), color=(32, 64, 128))
        intrinsics = torch.tensor(
            [[600.0, 0.0, 40.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]]
        )

        depth = model.predict(image, intrinsics=intrinsics)

        assert depth.shape == (64, 80)
        assert depth.dtype == next(model.parameters()).dtype
        assert torch.isfinite(depth).all()

    @pytest.mark.parametrize("model_name", _NON_FOCAL_NAMES)
    def test_predict_batch__mixed_sizes(self, model_name: str) -> None:
        model = _build(model_name)
        images = [
            Image.new("RGB", (80, 64), color=(32, 64, 128)),
            Image.new("RGB", (56, 56), color=(128, 64, 32)),
        ]

        depths = model.predict_batch(images)

        # The processed sizes differ, so the batch is center-cropped to the smallest
        # processed size before inference; each depth map is still resized back to its
        # image's original resolution.
        assert len(depths) == 2
        assert depths[0].shape == (64, 80)
        assert depths[1].shape == (56, 56)
        for depth in depths:
            assert torch.isfinite(depth).all()

    def test_predict__max_depth_scales_output(self) -> None:
        # The metric depth scales linearly with max_depth: with identical weights,
        # doubling max_depth must exactly double the predicted depth.
        model_20 = _build("dinov2/dav2-metric-large-hypersim", max_depth=20.0)
        model_40 = _build("dinov2/dav2-metric-large-hypersim", max_depth=40.0)
        model_40.load_state_dict(model_20.state_dict())
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))

        depth_20 = model_20.predict(image)
        depth_40 = model_40.predict(image)

        assert torch.equal(depth_40, depth_20 * 2.0)

    def test_predict__intrinsics_scale_depth_by_focal(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        # A 56x56 image is processed at its own size, so focal lengths are not rescaled.
        # fx = fy = 600 → scale 600/300 = 2; fx = fy = 1200 → scale 1200/300 = 4.
        # Doubling the focal should exactly double the metric depth.
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))
        intrinsics_600 = torch.tensor(
            [[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]
        )
        intrinsics_1200 = torch.tensor(
            [[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]
        )

        depth_600 = model.predict(image, intrinsics=intrinsics_600)
        depth_1200 = model.predict(image, intrinsics=intrinsics_1200)

        assert torch.equal(depth_1200, depth_600 * 2.0)

    def test_predict_batch__intrinsics_per_image(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        images = [
            Image.new("RGB", (56, 56), color=(32, 64, 128)),
            Image.new("RGB", (56, 56), color=(128, 64, 32)),
        ]
        intrinsics = [
            torch.tensor([[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]),
        ]
        # Scale each image's focal by 2 to verify per-image scaling.
        intrinsics_doubled = [
            torch.tensor([[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[2400.0, 0.0, 28.0], [0.0, 2400.0, 28.0], [0.0, 0.0, 1.0]]),
        ]

        metric = model.predict_batch(images, intrinsics=intrinsics)
        metric_doubled = model.predict_batch(images, intrinsics=intrinsics_doubled)

        assert torch.equal(metric_doubled[0], metric[0] * 2.0)
        assert torch.equal(metric_doubled[1], metric[1] * 2.0)

    def test_predict__intrinsics_required_for_focal_model(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))

        with pytest.raises(ValueError, match="requires per-image camera intrinsics"):
            model.predict(image)

    @pytest.mark.parametrize("model_name", _NON_FOCAL_NAMES)
    def test_predict__intrinsics_rejected_for_nonfocal_model(
        self, model_name: str
    ) -> None:
        model = _build(model_name)
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))
        intrinsics = torch.tensor(
            [[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]
        )

        with pytest.raises(ValueError, match="does not accept intrinsics"):
            model.predict(image, intrinsics=intrinsics)

    @pytest.mark.parametrize("model_name", _NO_SKY_NAMES)
    def test_forward__returns_depth(self, model_name: str) -> None:
        model = _build(model_name)
        x = torch.randn(2, 3, 56, 70)

        out = model(x)

        # Depth Anything V2 has no sky head, so the forward output is depth only.
        assert out["depth"].shape == (2, 1, 56, 70)
        assert "sky" not in out

    @pytest.mark.parametrize("model_name", _SKY_NAMES)
    def test_forward__returns_depth_and_sky(self, model_name: str) -> None:
        model = _build(model_name)
        x = torch.randn(2, 3, 56, 70)

        out = model(x)

        assert out["depth"].shape == (2, 1, 56, 70)
        assert out["sky"].shape == (2, 1, 56, 70)


def test__processed_focal_length() -> None:
    intrinsics = torch.tensor(
        [[400.0, 0.0, 100.0], [0.0, 300.0, 50.0], [0.0, 0.0, 1.0]]
    )

    focal = task_model._processed_focal_length(
        intrinsics=intrinsics,
        orig_h=100,
        orig_w=200,
        proc_h=50,
        proc_w=100,
    )

    # fx scales by the width ratio (400 * 0.5 = 200), fy by the height ratio
    # (300 * 0.5 = 150), and the focal is their average.
    assert focal == 175.0
