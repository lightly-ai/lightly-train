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

from lightly_train._task_models.dinov2_dav2_relative_depth_estimation.task_model import (
    DepthAnythingV2RelativeDepthEstimation,
)


@pytest.fixture()
def tiny_model_args() -> dict[str, Any]:
    return {
        "out_layers": (0, 1, 2, 3),
        "image_size": 56,
        "patch_size": 14,
        "features": 16,
        "out_channels": (8, 16, 32, 32),
    }


class TestDepthAnythingV2RelativeDepthEstimation:
    def test_predict__returns_original_resolution(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        model = DepthAnythingV2RelativeDepthEstimation(
            model_name="dinov2/dav2-relative-large",
            model_args=tiny_model_args,
            load_weights=False,
        )
        # Production fixes the inference size at 518; override it here so inference
        # runs at a tiny resolution and the test stays fast.
        model.inference_size = 56
        image = Image.new("RGB", (80, 64), color=(32, 64, 128))

        depth = model.predict(image)

        # Inference runs at the processing resolution (56, 70) and the depth map is
        # resized back to the original input resolution.
        assert depth.shape == (64, 80)
        assert depth.dtype == next(model.parameters()).dtype
        assert torch.isfinite(depth).all()

    def test_predict_batch__mixed_sizes(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        model = DepthAnythingV2RelativeDepthEstimation(
            model_name="dinov2/dav2-relative-large",
            model_args=tiny_model_args,
            load_weights=False,
        )
        # Production fixes the inference size at 518; override it here so inference
        # runs at a tiny resolution and the test stays fast.
        model.inference_size = 56
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

    def test_forward__returns_depth(
        self,
        tiny_model_args: dict[str, Any],
    ) -> None:
        model = DepthAnythingV2RelativeDepthEstimation(
            model_name="dinov2/dav2-relative-large",
            model_args=tiny_model_args,
            load_weights=False,
        )
        # Production fixes the inference size at 518; override it here so inference
        # runs at a tiny resolution and the test stays fast.
        model.inference_size = 56
        x = torch.randn(2, 3, 56, 70)

        out = model(x)

        # Depth Anything V2 has no sky head, so the forward output is depth only.
        assert out["depth"].shape == (2, 1, 56, 70)
        assert "sky" not in out
