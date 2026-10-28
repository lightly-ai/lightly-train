#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lightly_train._models import package_helpers
from lightly_train._models.dinov3.dinov3_src.layers.rope_position_encoding import (
    RopePositionEmbedding,
)
from lightly_train._models.lightlyvit import (
    LIGHTLYVIT_PACKAGE,
    LightlyViT,
    LightlyViTBackboneOutput,
    lightlyvit_l16_v1,
    lightlyvit_m16_v1,
    lightlyvit_s16_v1,
)
from lightly_train._models.lightlyvit.lightlyvit_mixers import (
    apply_rope_to_qk,
    rope_apply,
)


def _tiny_model(
    pos_encoding: str = "learned_absolute",
    global_mixer_kind: str = "softmax",
) -> LightlyViT:
    return LightlyViT(
        image_size=(32, 48),
        embed_dim=24,
        depth=4,
        num_heads=3,
        num_register_tokens=2,
        global_block_indices=(1, 3),
        pos_encoding=pos_encoding,
        global_mixer_kind=global_mixer_kind,
    )


def test_public_architecture() -> None:
    model = lightlyvit_s16_v1()
    assert model.embed_dim == 432
    assert model.num_blocks == 12
    assert model.num_heads == 9
    assert model.num_register_tokens == 4
    assert model.global_block_indices == (3, 7, 11)
    assert model.global_mixer_kind == "softmax"
    assert (
        20_600_000
        < sum(parameter.numel() for parameter in model.parameters())
        < 20_900_000
    )


def test_public_architecture_medium() -> None:
    model = lightlyvit_m16_v1()
    assert model.embed_dim == 640
    assert model.num_blocks == 16
    assert model.num_heads == 10
    assert model.num_register_tokens == 4
    assert model.global_block_indices == (3, 7, 11, 15)
    # M/16 defaults to the gate-winning linear global mixer.
    assert model.global_mixer_kind == "linear"
    assert (
        58_000_000
        < sum(parameter.numel() for parameter in model.parameters())
        < 62_000_000
    )


def test_public_architecture_large() -> None:
    model = lightlyvit_l16_v1()
    assert model.embed_dim == 1024
    assert model.num_blocks == 32
    assert model.num_heads == 16
    assert model.num_register_tokens == 4
    assert model.global_block_indices == tuple(range(3, 32, 4))
    assert model.global_mixer_kind == "linear"
    params = sum(parameter.numel() for parameter in model.parameters())
    assert 300_000_000 < params < 310_000_000


@pytest.mark.parametrize(
    "shape, grid", [((32, 48), (2, 3)), ((31, 35), (2, 3)), ((17, 16), (2, 1))]
)
def test_forward_metadata_shapes_gradients(
    shape: tuple[int, int], grid: tuple[int, int]
) -> None:
    model = _tiny_model()
    x = torch.randn(2, 3, *shape, requires_grad=True)
    output = model.forward_features(x)
    assert isinstance(output, LightlyViTBackboneOutput)
    assert output.features.shape == (2, 24, *grid)
    assert output.patch_tokens.shape == (2, grid[0] * grid[1], 24)
    assert output.grid_metadata.original_size == shape
    assert output.grid_metadata.grid_size == grid
    assert output.valid_token_mask.shape == (2, *grid)
    assert output.valid_token_mask.all()
    output.features.sum().backward()
    assert x.grad is not None


def test_ssl_masks_use_padded_ceil_grid() -> None:
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 31, 35)
    masks = torch.tensor([[[False, True, False], [True, False, True]]])
    output = model.forward_features(x, masks=masks)
    assert output.grid_metadata.grid_size == (2, 3)
    assert output.valid_token_mask.all()
    with pytest.raises(ValueError, match="padded patch grid"):
        model.forward_features(x, masks=torch.zeros(1, 1, 2, dtype=torch.bool))


def test_intermediate_layers_validate_and_preserve_order() -> None:
    model = _tiny_model().eval()
    x = torch.randn(1, 3, 31, 35)
    outputs = model.get_intermediate_layers(
        x,
        n=[3, 1, 3],
        reshape=True,
        return_class_token=True,
        return_extra_tokens=True,
    )
    assert [tuple(item.shape) for item in outputs[0]] == [
        (1, 24, 2, 3),
        (1, 24),
        (1, 2, 24),
    ]
    torch.testing.assert_close(outputs[0][0], outputs[2][0])
    with pytest.raises(ValueError, match="Layer indices"):
        model.get_intermediate_layers(x, n=[4])
    with pytest.raises(ValueError, match="Layer indices"):
        model.forward_intermediate_features(x, [-1])


def test_local_spatial_mixer_does_not_touch_special_prefix() -> None:
    model = _tiny_model().eval()
    block = model.blocks[0]
    block.ls2.gamma.data.zero_()
    tokens = torch.randn(1, 7, 24)
    output = block(tokens, (1, 3), num_special_tokens=4)
    torch.testing.assert_close(output[:, :4], tokens[:, :4])
    assert not torch.equal(output[:, 4:], tokens[:, 4:])


def test_global_attention_can_update_special_prefix() -> None:
    model = _tiny_model().eval()
    block = model.blocks[1]
    block.ls2.gamma.data.zero_()
    block.ls1.gamma.data.fill_(1.0)
    tokens = torch.randn(1, 7, 24)
    output = block(tokens, (1, 3), num_special_tokens=4)
    assert not torch.equal(output[:, :4], tokens[:, :4])


@pytest.mark.parametrize(
    "model_name, embed_dim",
    [
        ("lightlyvit_t16_v1", 224),
        ("lightlyvit_s16_v1", 432),
        ("lightlyvit_m16_v1", 640),
        ("lightlyvit_l16_v1", 1024),
    ],
)
def test_wrapper_discovery_pool_and_state_dict_roundtrip(
    tmp_path: Path, model_name: str, embed_dim: int
) -> None:
    model = LIGHTLYVIT_PACKAGE.get_model(
        model_name, model_args={"image_size": (32, 48)}
    )
    wrapper = LIGHTLYVIT_PACKAGE.get_model_wrapper(model)
    features = wrapper.forward_features(torch.randn(1, 3, 32, 48))
    pooled = wrapper.forward_pool(features)
    assert features["features"].shape == (1, embed_dim, 2, 3)
    assert pooled["pooled_features"].shape == (1, embed_dim, 1, 1)
    assert wrapper.architecture_info() == {
        "model_type": "hybrid",
        "norm_type": "layernorm",
    }
    assert package_helpers.get_package("lightlyvit") is LIGHTLYVIT_PACKAGE
    assert f"lightlyvit/{model_name}" in package_helpers.list_model_names()

    path = tmp_path / "backbone.pt"
    LIGHTLYVIT_PACKAGE.export_model(wrapper, path, log_example=False)
    loaded = LIGHTLYVIT_PACKAGE.get_model(
        model_name,
        model_args={"image_size": (32, 48), "weights": path},
    )
    for expected, actual in zip(
        model.state_dict().values(), loaded.state_dict().values()
    ):
        torch.testing.assert_close(actual, expected)


def test_rope_rotation_preserves_norms_and_prefix() -> None:
    rope = RopePositionEmbedding(embed_dim=24, num_heads=3)
    sin, cos = rope(H=3, W=4)
    assert sin.shape == (12, 8)
    assert cos.shape == (12, 8)
    q = torch.randn(2, 3, 16, 8)  # B x heads x (4 prefix + 12 patch) x head_dim
    k = torch.randn(2, 3, 16, 8)
    q_rot, k_rot = apply_rope_to_qk(q=q, k=k, rope=(sin, cos))
    torch.testing.assert_close(q_rot[:, :, :4], q[:, :, :4])
    torch.testing.assert_close(k_rot[:, :, :4], k[:, :, :4])
    # Rotations are orthogonal: per-token vector norms are preserved.
    torch.testing.assert_close(
        q_rot[:, :, 4:].norm(dim=-1), q[:, :, 4:].norm(dim=-1), rtol=1e-4, atol=1e-5
    )
    with pytest.raises(ValueError, match="RoPE table covers"):
        apply_rope_to_qk(q=q[:, :, :8], k=k[:, :, :8], rope=(sin, cos))


def test_rope_scores_depend_on_relative_offsets() -> None:
    torch.manual_seed(0)
    rope = RopePositionEmbedding(embed_dim=24, num_heads=3)
    sin, cos = rope(H=3, W=4)
    q = torch.randn(8)
    k = torch.randn(8)

    def rotated_score(
        position_a: tuple[int, int], position_b: tuple[int, int]
    ) -> torch.Tensor:
        index_a = position_a[0] * 4 + position_a[1]
        index_b = position_b[0] * 4 + position_b[1]
        q_rot = rope_apply(
            q.unsqueeze(0), sin[index_a : index_a + 1], cos[index_a : index_a + 1]
        )
        k_rot = rope_apply(
            k.unsqueeze(0), sin[index_b : index_b + 1], cos[index_b : index_b + 1]
        )
        return (q_rot * k_rot).sum()

    # For FIXED query/key content, the rotated similarity depends only on the
    # relative offset: equal row offsets (any column) and equal column offsets
    # (any row) give equal scores.
    row_offset_scores = [
        rotated_score((row, column), (row + 1, column))
        for row in range(2)
        for column in range(4)
    ]
    for score in row_offset_scores[1:]:
        torch.testing.assert_close(score, row_offset_scores[0], rtol=1e-4, atol=1e-5)
    column_offset_scores = [
        rotated_score((row, column), (row, column + 1))
        for row in range(3)
        for column in range(3)
    ]
    for score in column_offset_scores[1:]:
        torch.testing.assert_close(score, column_offset_scores[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("global_mixer_kind", ["softmax", "linear"])
def test_rope_model_forward_state_dict_and_gradients(
    global_mixer_kind: str,
) -> None:
    model = _tiny_model(pos_encoding="rope", global_mixer_kind=global_mixer_kind)
    x = torch.randn(2, 3, 32, 48, requires_grad=True)
    output = model.forward_features(x)
    assert output.features.shape == (2, 24, 2, 3)
    output.features.sum().backward()  # type: ignore[no-untyped-call]
    assert x.grad is not None

    state = model.state_dict()
    assert "rope_embed.periods" in state
    assert "pos_embed" not in state

    # Off-base rectangular and padded grids need no interpolation in rope mode.
    assert model.forward_features(torch.randn(1, 3, 16, 64)).features.shape == (
        1,
        24,
        1,
        4,
    )
    masked = model.forward_features(
        torch.randn(1, 3, 31, 35), masks=torch.zeros(1, 2, 3, dtype=torch.bool)
    )
    assert masked.features.shape == (1, 24, 2, 3)


def test_rope_rescale_coords_active_only_in_training() -> None:
    model = _tiny_model(pos_encoding="rope")
    model.eval()
    torch.manual_seed(1)
    sin_eval_a, _ = model.rope_embed(H=2, W=3)
    torch.manual_seed(2)
    sin_eval_b, _ = model.rope_embed(H=2, W=3)
    torch.testing.assert_close(sin_eval_a, sin_eval_b)
    model.train()
    torch.manual_seed(3)
    sin_train, _ = model.rope_embed(H=2, W=3)
    assert not torch.allclose(sin_eval_a, sin_train)


def test_rope_replaces_patch_pos_embed_table() -> None:
    learned = lightlyvit_s16_v1()
    rope = LIGHTLYVIT_PACKAGE.get_model(
        "lightlyvit_s16_v1",
        model_args={"pos_encoding": "rope"},
    )
    learned_params = sum(parameter.numel() for parameter in learned.parameters())
    rope_params = sum(parameter.numel() for parameter in rope.parameters())
    # Exactly the base-grid patch table is removed; the rope module has no
    # parameters (a 12-entry fp32 buffer does not count).
    assert learned_params - rope_params == 432 * 14 * 14
    assert "rope_embed.periods" in rope.state_dict()


def test_rope_requires_divisible_head_dim() -> None:
    with pytest.raises(ValueError, match="4 \\* num_heads"):
        LightlyViT(
            image_size=(32, 48),
            embed_dim=45,
            depth=2,
            num_heads=9,
            global_block_indices=(1,),
            pos_encoding="rope",
        )


def test_interpolate_pos_encoding_rejected_in_rope_mode() -> None:
    model = _tiny_model(pos_encoding="rope")
    with pytest.raises(RuntimeError, match="learned_absolute"):
        model.interpolate_pos_encoding((2, 3))


def test_rope_package_variant_roundtrip(tmp_path: Path) -> None:
    model = LIGHTLYVIT_PACKAGE.get_model(
        "lightlyvit_s16_v1",
        model_args={"image_size": (32, 48), "pos_encoding": "rope"},
    )
    wrapper = LIGHTLYVIT_PACKAGE.get_model_wrapper(model)
    features = wrapper.forward_features(torch.randn(1, 3, 32, 48))
    assert features["features"].shape == (1, 432, 2, 3)

    path = tmp_path / "rope.pt"
    LIGHTLYVIT_PACKAGE.export_model(wrapper, path, log_example=False)
    loaded = LIGHTLYVIT_PACKAGE.get_model(
        "lightlyvit_s16_v1",
        model_args={"image_size": (32, 48), "pos_encoding": "rope", "weights": path},
    )
    for expected, actual in zip(
        model.state_dict().values(), loaded.state_dict().values()
    ):
        torch.testing.assert_close(actual, expected)
    # Cross-scheme loads fail loudly instead of silently dropping keys.
    with pytest.raises(RuntimeError):
        LIGHTLYVIT_PACKAGE.get_model(
            "lightlyvit_s16_v1",
            model_args={"image_size": (32, 48), "weights": path},
        )
