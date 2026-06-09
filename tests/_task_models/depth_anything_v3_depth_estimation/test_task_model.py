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

from lightly_train._task_models.dinov2_dav3_relative_depth_estimation.task_model import (
    DepthAnythingV3RelativeDepthEstimation,
)


@pytest.fixture()
def tiny_model_args() -> dict[str, Any]:
    return {
        "backbone_name": "custom",
        "out_layers": (0, 1, 2, 3),
        "image_size": 56,
        "patch_size": 14,
        "dim_in": 32,
        "features": 16,
        "out_channels": (8, 16, 32, 32),
        "backbone_args": {
            "embed_dim": 32,
            "depth": 4,
            "num_heads": 4,
        },
    }


def test_task_model__predict_returns_processed_resolution(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3RelativeDepthEstimation(
        model_name="dinov2/dav3-relative-large",
        process_resolution=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    image = Image.new("RGB", (80, 64), color=(32, 64, 128))

    depth = model.predict(image)

    # Matches the official DA3 `Prediction` resolution: the longest side (80) is
    # bounded to `process_resolution` (56), giving (W=56, H=45), then both sides are
    # rounded to the nearest multiple of the patch size (14), giving (W=56, H=42).
    assert depth.shape == (42, 56)
    assert depth.dtype == next(model.parameters()).dtype
    assert torch.isfinite(depth).all()


def test_task_model__forward_returns_depth_and_sky(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3RelativeDepthEstimation(
        model_name="dinov2/dav3-relative-large",
        process_resolution=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    x = torch.randn(2, 3, 56, 70)

    out = model(x)

    assert out["depth"].shape == (2, 1, 56, 70)
    assert out["sky"].shape == (2, 1, 56, 70)
