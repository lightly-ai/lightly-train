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


def test_interpolate_pos_embed_hook__bicubic_square_grid() -> None:
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


def test_interpolate_pos_embed_hook__bicubic_square_grid_no_prefix() -> None:
    # Direct load into the transformer: key is ``pos_embed`` (prefix="").
    cls = torch.randn(1, 1, 384)
    patches = torch.randn(1, 256, 384)
    state_dict = {"pos_embed": torch.cat([cls, patches], dim=1)}
    module = _module(torch.zeros(1, 1370, 384))
    _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
    assert state_dict["pos_embed"].shape == (1, 1370, 384)
    assert_close(state_dict["pos_embed"][:, :1], cls)


def test_interpolate_pos_embed_hook__noop_matching_shape() -> None:
    pos_embed = torch.randn(1, 1370, 384)
    state_dict = {"pos_embed": pos_embed}
    module = _module(torch.zeros(1, 1370, 384))
    _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
    # Unchanged when checkpoint and model grids already match.
    assert state_dict["pos_embed"] is pos_embed


def test_interpolate_pos_embed_hook__skip_nonsquare_grid() -> None:
    # 1 + 10 patches: 10 is not a perfect square -> left untouched.
    pos_embed = torch.randn(1, 11, 384)
    state_dict = {"pos_embed": pos_embed}
    module = _module(torch.zeros(1, 1370, 384))
    _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
    assert state_dict["pos_embed"] is pos_embed


def test_interpolate_pos_embed_hook__ignores_non_pos_embed_keys() -> None:
    weight = torch.randn(3, 384)
    state_dict = {"blocks.0.attn.qkv.weight": weight}
    module = _module(torch.zeros(1, 1370, 384))
    _model_helpers.interpolate_pos_embed_hook(module, state_dict, "")
    assert state_dict["blocks.0.attn.qkv.weight"] is weight


class _InChansModule(Module):
    """Minimal stand-in whose only attribute the input-channel hooks read."""

    def __init__(self, in_chans: int) -> None:
        super().__init__()
        self.in_chans = in_chans


def test_adjust_conv_input_channels__noop_same_channels() -> None:
    weight = torch.randn(8, 3, 3, 3)
    out = _model_helpers.adjust_conv_input_channels(weight, 3)
    assert torch.equal(out, weight)


def test_adjust_conv_input_channels__slice_fewer_channels() -> None:
    weight = torch.randn(8, 3, 3, 3)
    out = _model_helpers.adjust_conv_input_channels(weight, 1)
    assert out.shape == (8, 1, 3, 3)
    assert torch.equal(out, weight[:, :1])


def test_adjust_conv_input_channels__repeat_more_channels_with_remainder() -> None:
    weight = torch.randn(8, 3, 3, 3)
    out = _model_helpers.adjust_conv_input_channels(weight, 4)
    assert out.shape == (8, 4, 3, 3)
    assert torch.equal(out, torch.cat([weight, weight[:, :1]], dim=1))


def test_adjust_conv_input_channels__repeat_more_channels_exact_multiple() -> None:
    weight = torch.randn(8, 3, 3, 3)
    out = _model_helpers.adjust_conv_input_channels(weight, 6)
    assert out.shape == (8, 6, 3, 3)
    assert torch.equal(out, weight.repeat(1, 2, 1, 1))


def test_patch_embed_adjust_input_channels_hook__adjusts_proj_weight() -> None:
    weight = torch.randn(8, 3, 16, 16)
    state_dict = {"patch_embed.proj.weight": weight}
    module = _InChansModule(in_chans=5)
    _model_helpers.patch_embed_adjust_input_channels_hook(
        module, state_dict, "patch_embed."
    )
    assert state_dict["patch_embed.proj.weight"].shape == (8, 5, 16, 16)


def test_patch_embed_adjust_input_channels_hook__ignores_missing_key() -> None:
    weight = torch.randn(2, 2)
    state_dict = {"other.weight": weight}
    module = _InChansModule(in_chans=5)
    _model_helpers.patch_embed_adjust_input_channels_hook(
        module, state_dict, "patch_embed."
    )
    assert state_dict["other.weight"] is weight


def test_conv_pyramid_patch_embed_adjust_input_channels_hook__adjusts_first_conv_weight() -> (
    None
):
    weight = torch.randn(8, 3, 3, 3)
    state_dict = {"patch_embed.convs.0.conv.weight": weight}
    module = _InChansModule(in_chans=2)
    _model_helpers.conv_pyramid_patch_embed_adjust_input_channels_hook(
        module, state_dict, "patch_embed."
    )
    adjusted = state_dict["patch_embed.convs.0.conv.weight"]
    assert adjusted.shape == (8, 2, 3, 3)
    assert torch.equal(adjusted, weight[:, :2])
