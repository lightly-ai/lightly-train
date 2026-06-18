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

from lightly_train._task_models.dinov2_dav2_metric_depth_estimation.task_model import (
    DepthAnythingV2MetricDepthEstimation,
)


@pytest.fixture()
def tiny_model_args() -> dict[str, Any]:
    return {
        "out_layers": (0, 1, 2, 3),
        "image_size": 56,
        "patch_size": 14,
        "features": 16,
        "out_channels": (8, 16, 32, 32),
        "max_depth": 20.0,
    }


class TestDepthAnythingV2MetricDepthEstimation:
    def test_predict__returns_original_resolution(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        model = DepthAnythingV2MetricDepthEstimation(
            model_name="dinov2/dav2-metric-large-hypersim",
            process_resolution=56,
            model_args=tiny_model_args,
            load_weights=False,
        )
        image = Image.new("RGB", (80, 64), color=(32, 64, 128))

        depth = model.predict(image)

        # Inference runs at the processing resolution (56, 70) and the depth map is
        # resized back to the original input resolution.
        assert depth.shape == (64, 80)
        assert depth.dtype == next(model.parameters()).dtype
        assert torch.isfinite(depth).all()
        # The sigmoid head output in [0, 1] is scaled by max_depth (20 m).
        assert (depth >= 0.0).all()
        assert (depth <= 20.0).all()

    def test_predict_batch__mixed_sizes(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        model = DepthAnythingV2MetricDepthEstimation(
            model_name="dinov2/dav2-metric-large-hypersim",
            process_resolution=56,
            model_args=tiny_model_args,
            load_weights=False,
        )
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

    def test_predict__max_depth_scales_output(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        # The metric depth scales linearly with max_depth: with identical weights,
        # doubling max_depth must exactly double the predicted depth.
        model_20 = DepthAnythingV2MetricDepthEstimation(
            model_name="dinov2/dav2-metric-large-hypersim",
            process_resolution=56,
            model_args={**tiny_model_args, "max_depth": 20.0},
            load_weights=False,
        )
        model_40 = DepthAnythingV2MetricDepthEstimation(
            model_name="dinov2/dav2-metric-large-hypersim",
            process_resolution=56,
            model_args={**tiny_model_args, "max_depth": 40.0},
            load_weights=False,
        )
        model_40.load_state_dict(model_20.state_dict())
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))

        depth_20 = model_20.predict(image)
        depth_40 = model_40.predict(image)

        assert torch.equal(depth_40, depth_20 * 2.0)

    def test_forward__returns_depth(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        model = DepthAnythingV2MetricDepthEstimation(
            model_name="dinov2/dav2-metric-large-hypersim",
            process_resolution=56,
            model_args=tiny_model_args,
            load_weights=False,
        )
        x = torch.randn(2, 3, 56, 70)

        out = model(x)

        # Depth Anything V2 has no sky head, so the forward output is depth only.
        assert out["depth"].shape == (2, 1, 56, 70)
        assert "sky" not in out
