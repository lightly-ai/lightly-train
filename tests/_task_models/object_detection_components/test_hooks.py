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
from torch.nn import Linear, Module, Sequential

from lightly_train._task_models.object_detection_components import hooks
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)


@pytest.mark.parametrize("decoder_class", [DFINETransformer, RTDETRTransformerv2])
@pytest.mark.parametrize("checkpoint_classes", [2, 3, 5])
@pytest.mark.parametrize("num_denoising", [0, 4])
def test_score_head_reuse_or_reinit_hook__load_decoder(
    decoder_class: type[DFINETransformer] | type[RTDETRTransformerv2],
    checkpoint_classes: int,
    num_denoising: int,
) -> None:
    def create_decoder(num_classes: int) -> Module:
        return decoder_class(
            num_classes=num_classes,
            hidden_dim=32,
            num_queries=4,
            feat_channels=[32],
            feat_strides=[8],
            num_levels=1,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            num_denoising=num_denoising,
        )

    # Nest the decoders to exercise the state-dict prefix used by task models.
    source = Sequential(create_decoder(num_classes=checkpoint_classes))
    target = Sequential(create_decoder(num_classes=3))
    checkpoint = source.state_dict()
    for value in checkpoint.values():
        if value.is_floating_point():
            value.fill_(0.75)
    initial = {key: value.clone() for key, value in target.state_dict().items()}
    target.load_state_dict(checkpoint, strict=True)

    for key, actual in target.state_dict().items():
        if "score_head" in key and checkpoint_classes != 3:
            torch.testing.assert_close(actual, initial[key])
        elif "denoising_class_embed" in key and checkpoint_classes != 3:
            torch.testing.assert_close(actual[:-1], initial[key][:-1])
            # The last embedding is padding, not a dataset-specific class.
            torch.testing.assert_close(actual[-1], checkpoint[key][-1])
        else:
            torch.testing.assert_close(actual, checkpoint[key])


@pytest.mark.parametrize("checkpoint_classes", [2, 3, 5])
@pytest.mark.parametrize("bias", [False, True])
def test__reuse_or_reinit(checkpoint_classes: int, bias: bool) -> None:
    head = Linear(in_features=4, out_features=3, bias=bias, dtype=torch.float64)
    checkpoint = {"weight": torch.full((checkpoint_classes, 4), 0.75)}
    if bias:
        checkpoint["bias"] = torch.full((checkpoint_classes,), 0.75)
    initial = {key: value.clone() for key, value in head.state_dict().items()}
    adjusted = hooks._reuse_or_reinit(
        head_module=head,
        state_dict=checkpoint,
        weight_key="weight",
        bias_key="bias",
    )
    assert adjusted == (checkpoint_classes != 3)
    if adjusted:
        for key, value in checkpoint.items():
            torch.testing.assert_close(value, initial[key])
            assert not value.requires_grad
            assert value.data_ptr() != head.state_dict()[key].data_ptr()
    else:
        for value in checkpoint.values():
            assert torch.all(value == 0.75)


def test__reuse_or_reinit__missing_weight() -> None:
    head = Linear(in_features=4, out_features=3)
    checkpoint = {"unrelated": torch.ones(1)}
    assert not hooks._reuse_or_reinit(
        head_module=head,
        state_dict=checkpoint,
        weight_key="weight",
        bias_key="bias",
    )
    assert list(checkpoint) == ["unrelated"]
