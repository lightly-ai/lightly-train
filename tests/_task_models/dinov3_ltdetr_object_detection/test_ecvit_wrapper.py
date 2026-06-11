#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lightly_train._task_models.dinov3_ltdetr_object_detection.ecvit_wrapper import (
    ECVIT_PRESETS,
    ECViTWrapper,
)


class TestECViTWrapper:
    @pytest.mark.parametrize(
        ("name", "expected_channels"),
        [
            ("ecvitt", 192),
            ("ecvittplus", 256),
            ("ecvits", 256),
            ("ecvitsplus", 256),
        ],
    )
    def test_forward__all_presets_return_three_projected_levels(
        self, name: str, expected_channels: int
    ) -> None:
        model = ECViTWrapper(name=name, depth=1, interaction_indexes=[0])
        model.eval()

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        assert isinstance(outputs, tuple)
        assert len(outputs) == 3
        assert [output.shape for output in outputs] == [
            torch.Size([2, expected_channels, 4, 4]),
            torch.Size([2, expected_channels, 2, 2]),
            torch.Size([2, expected_channels, 1, 1]),
        ]

    def test_forward__averages_selected_layers_before_projecting(self) -> None:
        model = ECViTWrapper(
            name="ecvitt",
            embed_dim=16,
            num_heads=1,
            depth=2,
            interaction_indexes=[0, 1],
            proj_dim=16,
        )
        model.eval()
        image = torch.randn(2, 3, 32, 32)
        captured_projector_inputs: list[torch.Tensor] = []

        def capture_projector_input(
            module: torch.nn.Module, inputs: tuple[torch.Tensor]
        ) -> None:
            del module
            captured_projector_inputs.append(inputs[0].detach().clone())

        hook_handles = [
            projector.register_forward_pre_hook(capture_projector_input)
            for projector in model.projector
        ]
        try:
            with torch.no_grad():
                return_layers = model.backbone(image)
                outputs = model(image)
        finally:
            for hook_handle in hook_handles:
                hook_handle.remove()

        H_c = image.shape[2] // model.patch_size
        W_c = image.shape[3] // model.patch_size
        fused_feats = torch.mean(torch.stack(return_layers), dim=0)
        fused_feats = fused_feats.transpose(1, 2).contiguous().view(2, -1, H_c, W_c)
        expected_projector_inputs = [
            torch.nn.functional.interpolate(
                fused_feats,
                size=[int(H_c * (2 ** (1 - level))), int(W_c * (2 ** (1 - level)))],
                mode="bilinear",
                align_corners=False,
            )
            for level in range(model.num_levels)
        ]

        assert len(outputs) == 3
        assert len(captured_projector_inputs) == 3
        for captured_input, expected_input in zip(
            captured_projector_inputs, expected_projector_inputs
        ):
            assert torch.allclose(captured_input, expected_input)

    def test_load_weights_path__raises_on_missing_backbone_key(
        self, tmp_path: Path
    ) -> None:
        source = ECViTWrapper(
            name="ecvitt",
            embed_dim=16,
            num_heads=1,
            depth=1,
            interaction_indexes=[0],
            proj_dim=16,
        )
        state_dict = source.backbone.state_dict()
        state_dict.popitem()
        weights_path = tmp_path / "ecvitt_missing_key.pth"
        torch.save(state_dict, weights_path)

        with pytest.raises(RuntimeError, match="Missing key"):
            ECViTWrapper(
                name="ecvitt",
                weights_path=weights_path,
                embed_dim=16,
                num_heads=1,
                depth=1,
                interaction_indexes=[0],
                proj_dim=16,
            )

    def test_load_weights_path__raises_on_unexpected_backbone_key(
        self, tmp_path: Path
    ) -> None:
        source = ECViTWrapper(
            name="ecvitt",
            embed_dim=16,
            num_heads=1,
            depth=1,
            interaction_indexes=[0],
            proj_dim=16,
        )
        state_dict = source.backbone.state_dict()
        state_dict["unexpected.weight"] = torch.empty(1)
        weights_path = tmp_path / "ecvitt_unexpected_key.pth"
        torch.save(state_dict, weights_path)

        with pytest.raises(RuntimeError, match="Unexpected key"):
            ECViTWrapper(
                name="ecvitt",
                weights_path=weights_path,
                embed_dim=16,
                num_heads=1,
                depth=1,
                interaction_indexes=[0],
                proj_dim=16,
            )

    @pytest.mark.parametrize("container_key", ["state_dict", "model", "backbone"])
    def test_load_weights_path__unwraps_common_checkpoint_containers(
        self, tmp_path: Path, container_key: str
    ) -> None:
        source = ECViTWrapper(
            name="ecvitt",
            embed_dim=16,
            num_heads=1,
            depth=1,
            interaction_indexes=[0],
            proj_dim=16,
        )
        weights_path = tmp_path / "ecvitt.pth"
        torch.save({container_key: source.backbone.state_dict()}, weights_path)

        loaded = ECViTWrapper(
            name="ecvitt",
            weights_path=weights_path,
            embed_dim=16,
            num_heads=1,
            depth=1,
            interaction_indexes=[0],
            proj_dim=16,
        )

        for key, value in source.backbone.state_dict().items():
            assert torch.equal(loaded.backbone.state_dict()[key], value)

    def test_init__invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ECViT model name"):
            ECViTWrapper(name="unknown")

    def test_init__invalid_num_levels_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Only support num_levels=3"):
            ECViTWrapper(name="ecvitt", num_levels=4)

    @pytest.mark.skipif(
        os.environ.get("ECVIT_CHECKPOINT_DIR") is None,
        reason=(
            "Set ECVIT_CHECKPOINT_DIR to a directory with pre-downloaded "
            "EdgeCrafter checkpoints for optional local validation."
        ),
    )
    @pytest.mark.parametrize("name", sorted(ECVIT_PRESETS))
    def test_official_ecvit_checkpoint_loads_strictly(self, name: str) -> None:
        checkpoint_dir = Path(os.environ["ECVIT_CHECKPOINT_DIR"])
        checkpoint_path = checkpoint_dir / f"{name}.pth"

        model = ECViTWrapper(name=name, weights_path=checkpoint_path)

        assert len(model.backbone.state_dict()) > 0
