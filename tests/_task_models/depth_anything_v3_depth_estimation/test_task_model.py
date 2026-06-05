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
from torch import Tensor

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
        image_size=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    image = Image.new("RGB", (80, 64), color=(32, 64, 128))

    depth = model.predict(image)

    # Matches the official DA3 `Prediction` resolution: the longest side (80) is
    # bounded to `image_size` (56), giving (W=56, H=45), then both sides are rounded
    # to the nearest multiple of the patch size (14), giving (W=56, H=42).
    assert depth.shape == (42, 56)
    assert depth.dtype == next(model.parameters()).dtype
    assert torch.isfinite(depth).all()


def test_task_model__forward_returns_depth_and_sky(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3RelativeDepthEstimation(
        model_name="dinov2/dav3-relative-large",
        image_size=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    x = torch.randn(2, 3, 56, 70)

    out = model(x)

    assert out["depth"].shape == (2, 1, 56, 70)
    assert out["sky"].shape == (2, 1, 56, 70)


def test_task_model__loads_state_dict_with_upstream_hf_layout(
    tiny_model_args: dict[str, Any],
) -> None:
    """Loads a state dict in the official HF ``model.safetensors`` key layout.

    Keys are prefixed ``model.backbone.pretrained.*`` / ``model.head.*`` and
    ``mask_token`` is absent (it only exists during MIM pretraining). Values are
    randomized so that a regression in the load path that silently skipped
    parameters would be caught.
    """
    model = DepthAnythingV3RelativeDepthEstimation(
        model_name="dinov2/dav3-relative-large",
        image_size=56,
        model_args=tiny_model_args,
        load_weights=False,
    )

    hf_layout_state_dict: dict[str, Tensor] = {}
    for key, value in model.state_dict().items():
        if key == "backbone.mask_token":
            continue
        if key.startswith("backbone."):
            hf_key = f"model.backbone.pretrained.{key[len('backbone.') :]}"
        else:
            assert key.startswith("head.")
            hf_key = f"model.{key}"
        hf_layout_state_dict[hf_key] = torch.randn_like(value)

    before = model.head.scratch.layer1_rn.weight.detach().clone()
    model.load_train_state_dict(hf_layout_state_dict)
    after = model.head.scratch.layer1_rn.weight.detach()

    assert not torch.allclose(before, after)
    assert torch.allclose(
        after,
        hf_layout_state_dict["model.head.scratch.layer1_rn.weight"],
    )


def test_task_model__unsupported_model_name_fails() -> None:
    with pytest.raises(ValueError, match="not supported"):
        DepthAnythingV3RelativeDepthEstimation(
            model_name="depth-anything-v3/da3-large",
            load_weights=False,
        )
