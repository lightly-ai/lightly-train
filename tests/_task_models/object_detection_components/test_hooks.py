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
from torch.nn import Module, Sequential

from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)


def _create_decoder(
    decoder_class: type[DFINETransformer] | type[RTDETRTransformerv2],
    num_classes: int,
) -> Module:
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
        num_denoising=4,
    )


def _load_checkpoint(
    decoder_class: type[DFINETransformer] | type[RTDETRTransformerv2],
    checkpoint_classes: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # Nest the decoders to exercise the state dict prefix used by task models.
    source = Sequential(_create_decoder(decoder_class, num_classes=checkpoint_classes))
    target = Sequential(_create_decoder(decoder_class, num_classes=3))
    checkpoint = source.state_dict()
    for value in checkpoint.values():
        if value.is_floating_point():
            value.fill_(0.5)
    initial = {key: value.clone() for key, value in target.state_dict().items()}
    target.load_state_dict(checkpoint, strict=True)
    return checkpoint, initial, target.state_dict()


@pytest.mark.parametrize("decoder_class", [DFINETransformer, RTDETRTransformerv2])
@pytest.mark.parametrize("checkpoint_classes", [2, 3, 5])
def test_score_head_reuse_or_reinit_hook(
    decoder_class: type[DFINETransformer] | type[RTDETRTransformerv2],
    checkpoint_classes: int,
) -> None:
    checkpoint, initial, loaded = _load_checkpoint(decoder_class, checkpoint_classes)
    score_head_keys = [key for key in loaded if "score_head" in key]
    assert score_head_keys
    for key in score_head_keys:
        expected = initial[key] if checkpoint_classes != 3 else checkpoint[key]
        torch.testing.assert_close(loaded[key], expected)
    # Other weights are loaded from the checkpoint.
    torch.testing.assert_close(
        loaded["0.query_pos_head.layers.0.weight"],
        checkpoint["0.query_pos_head.layers.0.weight"],
    )


@pytest.mark.parametrize("decoder_class", [DFINETransformer, RTDETRTransformerv2])
@pytest.mark.parametrize("checkpoint_classes", [2, 3, 5])
def test_denoising_class_embed_reuse_or_reinit_hook(
    decoder_class: type[DFINETransformer] | type[RTDETRTransformerv2],
    checkpoint_classes: int,
) -> None:
    checkpoint, initial, loaded = _load_checkpoint(decoder_class, checkpoint_classes)
    key = "0.denoising_class_embed.weight"
    expected = initial[key] if checkpoint_classes != 3 else checkpoint[key]
    torch.testing.assert_close(loaded[key], expected)
