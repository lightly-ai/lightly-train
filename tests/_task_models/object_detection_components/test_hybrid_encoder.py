#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)

_COMMON_KWARGS = dict(
    in_channels=[8, 8, 8],
    feat_strides=[8, 8, 8],
    hidden_dim=8,
    use_encoder_idx=[2],
    num_encoder_layers=1,
    nhead=1,
    dim_feedforward=32,
    dropout=0.0,
    enc_act="gelu",
    expansion=1.0,
    depth_mult=1.0,
    act="silu",
)


def test_state_dict_ignore_keys_drops_unused_downsample_convs() -> None:
    # Simulates a legacy checkpoint that carries real (but unused) downsample_convs
    # weights, produced by a HybridEncoder built with upsample=True.
    legacy_shaped = HybridEncoder(upsample=True, **_COMMON_KWARGS)  # type: ignore[arg-type]
    legacy_state_dict = legacy_shaped.state_dict()
    assert any(k.startswith("downsample_convs.") for k in legacy_state_dict)

    target = HybridEncoder(
        upsample=False,
        state_dict_ignore_keys={"downsample_convs."},
        **_COMMON_KWARGS,
    )
    # Would raise RuntimeError (unexpected keys) without state_dict_ignore_keys.
    target.load_state_dict(legacy_state_dict, strict=True)


def test_forward_with_upsample_false_and_same_resolution_taps() -> None:
    encoder = HybridEncoder(
        upsample=False,
        state_dict_ignore_keys={"downsample_convs."},
        **_COMMON_KWARGS,
    )
    feats = [torch.randn(1, 8, 4, 4) for _ in range(3)]
    outs = encoder(feats)

    assert len(outs) == 3
    for out in outs:
        assert out.shape == (1, 8, 4, 4)
