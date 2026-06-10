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
            depth=1,
            interaction_indexes=[0],
            proj_dim=16,
        )
        model.eval()

        layer_1 = torch.full((2, 4, 16), 2.0)
        layer_2 = torch.full((2, 4, 16), 4.0)

        def backbone_forward(x: torch.Tensor) -> list[torch.Tensor]:
            return [layer_1.to(device=x.device), layer_2.to(device=x.device)]

        model.backbone.forward = backbone_forward  # type: ignore[method-assign]

        captured_inputs: list[torch.Tensor] = []

        class CaptureProjector(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                captured_inputs.append(x.detach().clone())
                return x

        model.projector = torch.nn.ModuleList(
            [CaptureProjector(), CaptureProjector(), CaptureProjector()]
        )

        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 32, 32))

        assert len(outputs) == 3
        assert torch.allclose(captured_inputs[1], torch.full((2, 16, 2, 2), 3.0))

    def test_load_weights_path__loads_backbone_state_dict_strictly(
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
        weights_path = tmp_path / "ecvitt.pth"
        torch.save(source.backbone.state_dict(), weights_path)

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
        reason="Set ECVIT_CHECKPOINT_DIR to validate official EdgeCrafter checkpoints.",
    )
    @pytest.mark.parametrize("name", sorted(ECVIT_PRESETS))
    def test_official_ecvit_checkpoint_loads_strictly(self, name: str) -> None:
        checkpoint_dir = Path(os.environ["ECVIT_CHECKPOINT_DIR"])
        checkpoint_path = checkpoint_dir / f"{name}.pth"

        model = ECViTWrapper(name=name, weights_path=checkpoint_path)

        assert len(model.backbone.state_dict()) > 0
