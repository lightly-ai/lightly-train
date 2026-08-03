#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torch.nn import Module
from torch.testing import assert_close

from lightly_train._models import _model_helpers


class _PosEmbedModule(Module):
    """A minimal stand-in for a DinoVisionTransformer: the hook only reads
    ``module.pos_embed`` for the target shape."""

    def __init__(self, pos_embed: torch.Tensor) -> None:
        super().__init__()
        self.pos_embed = pos_embed


def _module(pos_embed: torch.Tensor) -> _PosEmbedModule:
    return _PosEmbedModule(pos_embed=pos_embed)


class TestInterpolatePosEmbedHook:
    def test__bicubic_square_grid(self) -> None:
        # 224px DINOv2 vits14 ckpt (1 cls + 16x16 grid) -> 518px model (37x37),
        # loaded through the wrapper so the key is prefixed with ``_model.``.
        cls = torch.randn(1, 1, 384)
        patches = torch.randn(1, 256, 384)
        state_dict = {"_model.pos_embed": torch.cat([cls, patches], dim=1)}
        module = _module(torch.zeros(1, 1370, 384))
        _model_helpers.interpolate_pos_embed_hook(module, state_dict, "_model.")
        interpolated = state_dict["_model.pos_embed"]
        assert interpolated.shape == (1, 1370, 384)
        # The cls token is preserved exactly.
        assert_close(interpolated[:, :1], cls)

    def test__bicubic_square_grid_no_prefix(self) -> None:
        # Direct load into the transformer: key is ``pos_embed`` (prefix="").
        cls = torch.randn(1, 1, 384)
        patches = torch.randn(1, 256, 384)
        state_dict = {"pos_embed": torch.cat([cls, patches], dim=1)}
        module = _module(torch.zeros(1, 1370, 384))
        _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
        assert state_dict["pos_embed"].shape == (1, 1370, 384)
        assert_close(state_dict["pos_embed"][:, :1], cls)

    def test__noop_matching_shape(self) -> None:
        pos_embed = torch.randn(1, 1370, 384)
        state_dict = {"pos_embed": pos_embed}
        module = _module(torch.zeros(1, 1370, 384))
        _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
        # Unchanged when checkpoint and model grids already match.
        assert state_dict["pos_embed"] is pos_embed

    def test__skip_nonsquare_grid(self) -> None:
        # 1 + 10 patches: 10 is not a perfect square -> left untouched.
        pos_embed = torch.randn(1, 11, 384)
        state_dict = {"pos_embed": pos_embed}
        module = _module(torch.zeros(1, 1370, 384))
        _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
        assert state_dict["pos_embed"] is pos_embed

    def test__ignores_non_pos_embed_keys(self) -> None:
        weight = torch.randn(3, 384)
        state_dict = {"blocks.0.attn.qkv.weight": weight}
        module = _module(torch.zeros(1, 1370, 384))
        _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
        assert state_dict["blocks.0.attn.qkv.weight"] is weight
