#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from lightly_train._activation_checkpointing import (
    ActivationCheckpointingArgs,
    maybe_checkpoint,
)
from tests import helpers


class TestActivationCheckpointingArgs:
    def test_defaults(self) -> None:
        args = ActivationCheckpointingArgs()
        assert args.enabled is False
        assert args.every_n_blocks == 1

    def test_from_dict(self) -> None:
        args = ActivationCheckpointingArgs(enabled=True, every_n_blocks=3)
        assert args.enabled is True
        assert args.every_n_blocks == 3

    def test_every_n_blocks_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ActivationCheckpointingArgs(enabled=True, every_n_blocks=0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValueError):
            ActivationCheckpointingArgs(enabled=True, unknown_field=42)


class TestMaybeCheckpoint:
    def test_disabled(self) -> None:
        linear = nn.Linear(16, 16)
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            linear,
            x,
            use_activation_checkpointing=False,
            block_index=0,
            every_n_blocks=1,
        )
        y.sum().backward()
        assert x.grad is not None

    def test_enabled(self) -> None:
        linear = nn.Linear(16, 16)
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            linear,
            x,
            use_activation_checkpointing=True,
            block_index=0,
            every_n_blocks=1,
        )
        y.sum().backward()
        assert x.grad is not None

    def test_every_n_blocks_skips(self) -> None:
        linear = nn.Linear(16, 16)
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            linear,
            x,
            use_activation_checkpointing=True,
            block_index=1,
            every_n_blocks=2,
        )
        y.sum().backward()
        assert x.grad is not None

    def test_every_n_blocks_applies(self) -> None:
        linear = nn.Linear(16, 16)
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            linear,
            x,
            use_activation_checkpointing=True,
            block_index=2,
            every_n_blocks=2,
        )
        y.sum().backward()
        assert x.grad is not None

    def test_with_kwargs(self) -> None:
        class BlockWithKwargs(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
                return self.linear(x) * scale

        block = BlockWithKwargs()
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            block,
            x,
            scale=2.0,
            use_activation_checkpointing=True,
            block_index=0,
            every_n_blocks=1,
        )
        y.sum().backward()
        assert x.grad is not None

    def test_gradient_correctness(self) -> None:
        torch.manual_seed(0)
        linear = nn.Linear(16, 16)
        x = torch.randn(2, 16, requires_grad=True)

        y_ref = linear(x)
        y_ref.sum().backward()
        grad_ref = x.grad.clone()

        x.grad = None
        y_ckpt = maybe_checkpoint(
            linear,
            x,
            use_activation_checkpointing=True,
            block_index=0,
            every_n_blocks=1,
        )
        y_ckpt.sum().backward()

        assert torch.allclose(y_ref, y_ckpt)
        assert torch.allclose(grad_ref, x.grad)


class TestDINOv2ViTActivationCheckpointing:
    def test_forward_backward(self) -> None:
        from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=56,
            patch_size=14,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
            activation_checkpointing=True,
        )
        model.train()
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        out = model.forward_features(x)
        loss = out["x_norm_clstoken"].sum()
        loss.backward()
        assert x.grad is not None

    def test_numerical_equivalence(self) -> None:
        from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
            DinoVisionTransformer,
        )

        vit_kwargs = {
            "img_size": 56,
            "patch_size": 14,
            "embed_dim": 64,
            "depth": 4,
            "num_heads": 4,
            "mlp_ratio": 2.0,
        }
        torch.manual_seed(0)
        model_ref = DinoVisionTransformer(**vit_kwargs, activation_checkpointing=False)
        torch.manual_seed(0)
        model_ckpt = DinoVisionTransformer(**vit_kwargs, activation_checkpointing=True)
        model_ckpt.load_state_dict(model_ref.state_dict())
        model_ref.train()
        model_ckpt.train()

        x = torch.randn(2, 3, 56, 56)
        out_ref = model_ref.forward_features(x)
        out_ckpt = model_ckpt.forward_features(x)

        assert torch.allclose(
            out_ref["x_norm_clstoken"], out_ckpt["x_norm_clstoken"], atol=1e-5
        )

    def test_every_n_blocks(self) -> None:
        from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=56,
            patch_size=14,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
            activation_checkpointing=True,
            activation_checkpointing_every_n_blocks=2,
        )
        model.train()
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        out = model.forward_features(x)
        out["x_norm_clstoken"].sum().backward()
        assert x.grad is not None


class TestDINOv3ViTActivationCheckpointing:
    def test_forward_backward(self) -> None:
        from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=56,
            patch_size=14,
            embed_dim=64,
            depth=4,
            num_heads=4,
            ffn_ratio=2.0,
            activation_checkpointing=True,
        )
        model.train()
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        out = model.forward_features(x)
        loss = out["x_norm_clstoken"].sum()
        loss.backward()
        assert x.grad is not None

    def test_list_input(self) -> None:
        from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
            DinoVisionTransformer,
        )

        model = DinoVisionTransformer(
            img_size=56,
            patch_size=14,
            embed_dim=64,
            depth=4,
            num_heads=4,
            ffn_ratio=2.0,
            activation_checkpointing=True,
        )
        model.train()
        x_list = [
            torch.randn(2, 3, 56, 56, requires_grad=True),
            torch.randn(2, 3, 56, 56, requires_grad=True),
        ]
        out_list = model.forward_features_list(x_list, [None, None])
        loss = sum(o["x_norm_clstoken"].sum() for o in out_list)
        loss.backward()
        assert all(x.grad is not None for x in x_list)


class TestECViTActivationCheckpointing:
    def test_forward_backward(self) -> None:
        from lightly_train._models.ecvit.ecvit import VisionTransformer

        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=64,
            depth=4,
            num_heads=4,
            ffn_ratio=2.0,
            return_layers=[2, 3],
            activation_checkpointing=True,
        )
        model.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        outs, _ = model.forward_with_grid(x)
        loss = sum(o.sum() for o in outs)
        loss.backward()
        assert x.grad is not None


class TestPretrainActivationCheckpointing:
    def test_pretrain_with_activation_checkpointing(self, tmp_path: Path) -> None:
        from lightly_train._commands import train

        data = tmp_path / "data"
        helpers.create_images(image_dir=data, files=10)
        train.pretrain(
            out=tmp_path / "out",
            data=data,
            model="dinov2/_vittest14",
            method="dinov2",
            batch_size=2,
            num_workers=0,
            epochs=1,
            accelerator="cpu",
            activation_checkpointing={"enabled": True},
        )
        assert (tmp_path / "out" / "checkpoints" / "last.ckpt").exists()

    def test_pretrain_without_activation_checkpointing(self, tmp_path: Path) -> None:
        from lightly_train._commands import train

        data = tmp_path / "data"
        helpers.create_images(image_dir=data, files=10)
        train.pretrain(
            out=tmp_path / "out",
            data=data,
            model="torchvision/resnet18",
            method="simclr",
            batch_size=4,
            num_workers=0,
            epochs=1,
            accelerator="cpu",
        )
        assert (tmp_path / "out" / "checkpoints" / "last.ckpt").exists()
