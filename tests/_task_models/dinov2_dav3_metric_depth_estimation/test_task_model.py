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

from lightly_train._task_models.dinov2_dav3_metric_depth_estimation import task_model
from lightly_train._task_models.dinov2_dav3_metric_depth_estimation.task_model import (
    DepthAnythingV3MetricDepthEstimation,
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


def test_task_model__predict_returns_original_resolution(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3MetricDepthEstimation(
        model_name="dinov2/dav3-metric-large",
        process_resolution=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    image = Image.new("RGB", (80, 64), color=(32, 64, 128))

    depth = model.predict(image)

    # Inference runs at the processing resolution (56, 42) and the depth map is
    # resized back to the original input resolution.
    assert depth.shape == (64, 80)
    assert depth.dtype == next(model.parameters()).dtype
    assert torch.isfinite(depth).all()


def test_task_model__predict_intrinsics_scale_depth_by_focal(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3MetricDepthEstimation(
        model_name="dinov2/dav3-metric-large",
        process_resolution=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    # A 56x56 image is processed at its own size, so the focal length is not rescaled
    # and fx = fy = 600 gives an exact metric scale of 600 / 300 = 2.
    image = Image.new("RGB", (56, 56), color=(32, 64, 128))
    intrinsics = torch.tensor([[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]])

    canonical = model.predict(image)
    metric = model.predict(image, intrinsics=intrinsics)

    assert torch.equal(metric, canonical * 2.0)


def test_task_model__predict_batch_intrinsics_per_image(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3MetricDepthEstimation(
        model_name="dinov2/dav3-metric-large",
        process_resolution=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    images = [
        Image.new("RGB", (56, 56), color=(32, 64, 128)),
        Image.new("RGB", (56, 56), color=(128, 64, 32)),
    ]
    intrinsics = [
        torch.tensor([[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]),
    ]

    canonical = model.predict_batch(images)
    metric = model.predict_batch(images, intrinsics=intrinsics)

    # Each image is scaled by its own focal: 600 / 300 = 2 and 1200 / 300 = 4.
    assert torch.equal(metric[0], canonical[0] * 2.0)
    assert torch.equal(metric[1], canonical[1] * 4.0)


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
