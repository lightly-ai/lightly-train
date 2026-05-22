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

from lightly_train._task_models.depth_anything_v3_depth_estimation.model import (
    DepthAnythingV3MonoNet,
)
from lightly_train._task_models.depth_anything_v3_depth_estimation.task_model import (
    DepthAnythingV3MonocularDepthEstimation,
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


def test_depth_anything_v3_mono_net__forward(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3MonoNet(**tiny_model_args)
    x = torch.randn(2, 1, 3, 56, 70)

    out = model(x)

    assert set(out) == {"depth", "sky"}
    assert out["depth"].shape == (2, 1, 56, 70)
    assert out["sky"].shape == (2, 1, 56, 70)
    assert out["depth"].dtype == x.dtype
    assert torch.isfinite(out["depth"]).all()


def test_task_model__predict_returns_original_resolution(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3MonocularDepthEstimation(
        model_name="depth-anything/DA3MONO-LARGE",
        image_size=56,
        model_args=tiny_model_args,
        load_weights=False,
    )
    image = Image.new("RGB", (80, 64), color=(32, 64, 128))

    depth = model.predict(image)

    assert depth.shape == (64, 80)
    assert depth.dtype == next(model.parameters()).dtype
    assert torch.isfinite(depth).all()


def test_task_model__forward_returns_depth_and_sky(
    tiny_model_args: dict[str, Any],
) -> None:
    model = DepthAnythingV3MonocularDepthEstimation(
        model_name="da3mono-large",
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
    model = DepthAnythingV3MonocularDepthEstimation(
        model_name="depth-anything-v3/da3mono-large",
        image_size=56,
        model_args=tiny_model_args,
        load_weights=False,
    )

    hf_layout_state_dict: dict[str, Tensor] = {}
    for key, value in model.state_dict().items():
        if key == "model.backbone.pretrained.mask_token":
            continue
        hf_layout_state_dict[key] = torch.randn_like(value)

    before = model.model.head.scratch.layer1_rn.weight.detach().clone()
    model.load_train_state_dict(hf_layout_state_dict)
    after = model.model.head.scratch.layer1_rn.weight.detach()

    assert not torch.allclose(before, after)
    assert torch.allclose(
        after,
        hf_layout_state_dict["model.head.scratch.layer1_rn.weight"],
    )


def test_task_model__unsupported_model_name_fails() -> None:
    with pytest.raises(ValueError, match="not supported"):
        DepthAnythingV3MonocularDepthEstimation(
            model_name="depth-anything-v3/da3-large",
            load_weights=False,
        )
