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
from torch.nn import Embedding, Linear, Module, ModuleList

from lightly_train._task_models.object_detection_components.hooks import (
    denoising_class_embed_reuse_or_reinit_hook,
    score_head_reuse_or_reinit_hook,
)


class _Decoder(Module):
    def __init__(self, num_classes: int, num_layers: int = 2, hidden_dim: int = 4):
        super().__init__()
        self.enc_score_head = Linear(hidden_dim, num_classes)
        self.dec_score_head = ModuleList(
            [Linear(hidden_dim, num_classes) for _ in range(num_layers)]
        )
        self.denoising_class_embed = Embedding(
            num_classes + 1, hidden_dim, padding_idx=num_classes
        )


def _state_dict(num_classes: int) -> dict[str, torch.Tensor]:
    # Constant values make it obvious whether the checkpoint or the module init won.
    checkpoint = _Decoder(num_classes=num_classes)
    with torch.no_grad():
        for parameter in checkpoint.parameters():
            parameter.fill_(7.0)
    return {key: value.clone() for key, value in checkpoint.state_dict().items()}


def test_score_head_reuse_or_reinit_hook__same_num_classes() -> None:
    module = _Decoder(num_classes=3)
    state_dict = _state_dict(num_classes=3)

    score_head_reuse_or_reinit_hook(module, state_dict, prefix="")

    assert torch.equal(state_dict["enc_score_head.weight"], torch.full((3, 4), 7.0))
    assert torch.equal(state_dict["dec_score_head.0.weight"], torch.full((3, 4), 7.0))


@pytest.mark.parametrize("num_classes_checkpoint", [2, 80])
def test_score_head_reuse_or_reinit_hook__different_num_classes(
    num_classes_checkpoint: int,
) -> None:
    module = _Decoder(num_classes=3)
    state_dict = _state_dict(num_classes=num_classes_checkpoint)

    score_head_reuse_or_reinit_hook(module, state_dict, prefix="")

    for key in [
        "enc_score_head.weight",
        "enc_score_head.bias",
        "dec_score_head.0.weight",
        "dec_score_head.0.bias",
        "dec_score_head.1.weight",
        "dec_score_head.1.bias",
    ]:
        expected = module.state_dict()[key]
        assert state_dict[key].shape == expected.shape
        assert torch.equal(state_dict[key], expected)


def test_score_head_reuse_or_reinit_hook__no_score_head() -> None:
    state_dict = _state_dict(num_classes=80)
    before = {key: value.clone() for key, value in state_dict.items()}

    score_head_reuse_or_reinit_hook(Module(), state_dict, prefix="")

    assert all(torch.equal(state_dict[key], before[key]) for key in before)


def test_denoising_class_embed_reuse_or_reinit_hook__same_num_classes() -> None:
    module = _Decoder(num_classes=3)
    state_dict = _state_dict(num_classes=3)

    denoising_class_embed_reuse_or_reinit_hook(module, state_dict, prefix="")

    assert torch.equal(
        state_dict["denoising_class_embed.weight"], torch.full((4, 4), 7.0)
    )


@pytest.mark.parametrize("num_classes_checkpoint", [2, 80])
def test_denoising_class_embed_reuse_or_reinit_hook__different_num_classes(
    num_classes_checkpoint: int,
) -> None:
    module = _Decoder(num_classes=3)
    state_dict = _state_dict(num_classes=num_classes_checkpoint)

    denoising_class_embed_reuse_or_reinit_hook(module, state_dict, prefix="")

    weight = state_dict["denoising_class_embed.weight"]
    assert torch.equal(weight, module.denoising_class_embed.weight)
    # The padding row must stay zero so padding_idx keeps its meaning.
    assert torch.equal(weight[-1], torch.zeros(4))


def test_hooks__load_state_dict_succeeds_with_different_num_classes() -> None:
    module = _Decoder(num_classes=3)
    state_dict = _state_dict(num_classes=80)

    score_head_reuse_or_reinit_hook(module, state_dict, prefix="")
    denoising_class_embed_reuse_or_reinit_hook(module, state_dict, prefix="")

    module.load_state_dict(state_dict, strict=True)
