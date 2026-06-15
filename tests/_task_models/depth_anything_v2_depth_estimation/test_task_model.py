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
from PIL import Image

from lightly_train._task_models.dinov2_dav2_relative_depth_estimation.task_model import (
    DepthAnythingV2RelativeDepthEstimation,
)


@pytest.fixture(scope="module")
def model() -> DepthAnythingV2RelativeDepthEstimation:
    # The small variant at a tiny process resolution keeps the test light while still
    # exercising the real DINOv2 backbone and DPT head. The model is only read, so it
    # is shared across tests.
    model = DepthAnythingV2RelativeDepthEstimation(
        model_name="dinov2/dav2-relative-small",
        process_resolution=42,
        load_weights=False,
    )
    model.eval()
    return model


def test_task_model__predict_returns_original_resolution(
    model: DepthAnythingV2RelativeDepthEstimation,
) -> None:
    image = Image.new("RGB", (80, 64), color=(32, 64, 128))

    depth = model.predict(image)

    # Inference runs at the processed resolution and the depth map is resized back to
    # the original input resolution.
    assert depth.shape == (64, 80)
    assert depth.dtype == next(model.parameters()).dtype
    assert torch.isfinite(depth).all()


def test_task_model__predict_batch_mixed_sizes(
    model: DepthAnythingV2RelativeDepthEstimation,
) -> None:
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


def test_task_model__forward_returns_depth_only(
    model: DepthAnythingV2RelativeDepthEstimation,
) -> None:
    x = torch.randn(2, 3, 56, 70)

    out = model(x)

    # Depth Anything V2 has no sky head.
    assert set(out.keys()) == {"depth"}
    assert out["depth"].shape == (2, 1, 56, 70)


def test_task_model__list_model_names() -> None:
    assert DepthAnythingV2RelativeDepthEstimation.list_model_names() == [
        "dinov2/dav2-relative-small",
        "dinov2/dav2-relative-base",
        "dinov2/dav2-relative-large",
    ]


def test_task_model__is_supported_model() -> None:
    assert DepthAnythingV2RelativeDepthEstimation.is_supported_model(
        "dinov2/dav2-relative-base"
    )
    assert not DepthAnythingV2RelativeDepthEstimation.is_supported_model("nope")
