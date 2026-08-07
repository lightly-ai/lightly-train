#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import pytest
import torch
from torch import nn

from lightly_train._activation_checkpointing import (
    ActivationCheckpointingArgs,
    maybe_checkpoint,
)
from lightly_train._models.model_wrapper import SupportsActivationCheckpointing
from tests import helpers


class ActivationCheckpointable(Protocol):
    """Bare model interface needed by the checkpointing test helper."""

    _activation_checkpointing: bool
    _activation_checkpointing_every_n_blocks: int


class DinoForwardFeatures(Protocol):
    """Typed subset of the vendor DINO forward-features interface used in tests."""

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...


class DinoV3ForwardFeatures(DinoForwardFeatures, Protocol):
    """DINOv3-specific list-input variant of forward-features."""

    def forward_features_list(
        self,
        x_list: list[torch.Tensor],
        masks_list: list[torch.Tensor | None],
    ) -> list[dict[str, torch.Tensor]]: ...


class CountingBlock(nn.Module):
    """Linear block that records how many times its forward ran.

    Checkpointing is only observable as a *recompute*: asserting on gradients
    alone passes even when no checkpointing happens at all.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.num_forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.num_forward_calls += 1
        return cast(torch.Tensor, self.linear(x))


def enable_checkpointing(
    model: ActivationCheckpointable, every_n_blocks: int = 1
) -> None:
    """Configure a bare ViT the way its model wrapper would."""
    model._activation_checkpointing = True
    model._activation_checkpointing_every_n_blocks = every_n_blocks


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
            ActivationCheckpointingArgs.model_validate(
                {"enabled": True, "unknown_field": 42}
            )


class TestMaybeCheckpoint:
    def test_disabled(self) -> None:
        block = CountingBlock()
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            block,
            x,
            use_activation_checkpointing=False,
            block_index=0,
            every_n_blocks=1,
        )
        y.sum().backward()
        assert x.grad is not None
        # Without checkpointing the block runs once, with no recompute.
        assert block.num_forward_calls == 1

    def test_enabled(self) -> None:
        block = CountingBlock()
        x = torch.randn(2, 16, requires_grad=True)
        y = maybe_checkpoint(
            block,
            x,
            use_activation_checkpointing=True,
            block_index=0,
            every_n_blocks=1,
        )
        y.sum().backward()
        assert x.grad is not None
        # With checkpointing the block is recomputed during backward.
        assert block.num_forward_calls == 2

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
                return cast(torch.Tensor, self.linear(x)) * scale

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
        assert x.grad is not None
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
        assert x.grad is not None

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
        )
        enable_checkpointing(model)
        model.train()
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        out = cast(DinoForwardFeatures, model).forward_features(x)
        loss = out["x_norm_clstoken"].sum()
        torch.autograd.backward(loss)
        assert x.grad is not None

    def test_numerical_equivalence(self) -> None:
        from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
            DinoVisionTransformer,
        )

        torch.manual_seed(0)
        model_ref = DinoVisionTransformer(
            img_size=56,
            patch_size=14,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
        )
        torch.manual_seed(0)
        model_ckpt = DinoVisionTransformer(
            img_size=56,
            patch_size=14,
            embed_dim=64,
            depth=4,
            num_heads=4,
            mlp_ratio=2.0,
        )
        enable_checkpointing(model_ckpt)
        model_ckpt.load_state_dict(model_ref.state_dict())
        model_ref.train()
        model_ckpt.train()

        x = torch.randn(2, 3, 56, 56)
        out_ref = cast(DinoForwardFeatures, model_ref).forward_features(x)
        out_ckpt = cast(DinoForwardFeatures, model_ckpt).forward_features(x)

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
        )
        enable_checkpointing(model, every_n_blocks=2)
        model.train()
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        out = cast(DinoForwardFeatures, model).forward_features(x)
        torch.autograd.backward(out["x_norm_clstoken"].sum())
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
        )
        enable_checkpointing(model)
        model.train()
        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        out = cast(DinoForwardFeatures, model).forward_features(x)
        loss = out["x_norm_clstoken"].sum()
        torch.autograd.backward(loss)
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
        )
        enable_checkpointing(model)
        model.train()
        x_list = [
            torch.randn(2, 3, 56, 56, requires_grad=True),
            torch.randn(2, 3, 56, 56, requires_grad=True),
        ]
        out_list = cast(DinoV3ForwardFeatures, model).forward_features_list(
            x_list, [None, None]
        )
        loss = torch.stack([o["x_norm_clstoken"].sum() for o in out_list]).sum()
        torch.autograd.backward(loss)
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
        )
        enable_checkpointing(model)
        model.train()
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        outs, _ = model.forward_with_grid(x)
        loss = torch.stack([o.sum() for o in outs]).sum()
        torch.autograd.backward(loss)
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
            activation_checkpoint_args={"enabled": True},
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

    def test_pretrain_unsupported_model_raises(self, tmp_path: Path) -> None:
        from lightly_train._commands import train

        data = tmp_path / "data"
        helpers.create_images(image_dir=data, files=10)
        with pytest.raises(ValueError, match="not supported"):
            train.pretrain(
                out=tmp_path / "out",
                data=data,
                model="torchvision/resnet18",
                method="simclr",
                batch_size=4,
                num_workers=0,
                epochs=1,
                accelerator="cpu",
                activation_checkpoint_args={"enabled": True},
            )


class TestSupportsActivationCheckpointing:
    """Support is decided once, on the instantiated model wrapper."""

    def test_vit_wrappers_declare_support(self) -> None:
        from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
        from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
        from lightly_train._models.ecvit.ecvit import ECViTModelWrapper

        for wrapper_cls in (
            DINOv2ViTModelWrapper,
            DINOv3ViTModelWrapper,
            ECViTModelWrapper,
        ):
            assert issubclass(wrapper_cls, SupportsActivationCheckpointing)

    def test_non_vit_wrappers_do_not_declare_support(self) -> None:
        from lightly_train._models.dinov3.dinov3_convnext import (
            DINOv3VConvNeXtModelWrapper,
        )
        from lightly_train._models.torchvision.torchvision import (
            TorchvisionModelWrapper,
        )

        for wrapper_cls in (DINOv3VConvNeXtModelWrapper, TorchvisionModelWrapper):
            assert not issubclass(wrapper_cls, SupportsActivationCheckpointing)

    def test_dinov3_convnext_raises_value_error(self) -> None:
        """A ConvNeXt in the dinov3 package must not be treated as supported.

        Support used to be decided per-package, so dinov3/convnext passed the
        check and then failed with a raw TypeError from the model builder.
        """
        from lightly_train._commands import train_helpers
        from lightly_train._models import package_helpers

        wrapped_model = package_helpers.get_wrapped_model(
            model="dinov3/_convnexttest",
            num_input_channels=3,
            load_weights=False,
        )
        with pytest.raises(ValueError, match="not supported"):
            train_helpers.set_activation_checkpointing(
                wrapped_model=wrapped_model,
                args=ActivationCheckpointingArgs(enabled=True),
            )

    def test_instantiated_ecvit_wrapper_is_accepted(self) -> None:
        """An already-instantiated ECViT wrapper must be accepted.

        ECViTModelWrapper.get_model() returns self, so the previous structural
        check inspected the wrapper instead of the backbone and rejected it.
        """
        from lightly_train._commands import train_helpers
        from lightly_train._models import package_helpers
        from lightly_train._models.ecvit.ecvit import ECViTModelWrapper

        wrapped_model = package_helpers.get_wrapped_model(
            model="edgecrafter/ecvitt",
            num_input_channels=3,
            load_weights=False,
        )
        train_helpers.set_activation_checkpointing(
            wrapped_model=wrapped_model,
            args=ActivationCheckpointingArgs(enabled=True, every_n_blocks=2),
        )
        assert isinstance(wrapped_model, ECViTModelWrapper)
        backbone = wrapped_model.backbone_model
        assert cast(bool, backbone._activation_checkpointing) is True
        assert cast(int, backbone._activation_checkpointing_every_n_blocks) == 2


class TestBlockChunkNotDoubleCheckpointed:
    def test_chunked_blocks_recompute_once(self) -> None:
        """With block_chunks > 0 each block must recompute once, not twice.

        The outer loop over ``self.blocks`` already checkpoints whole chunks, so
        checkpointing again inside BlockChunk.forward would nest and recompute
        twice.
        """
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
            block_chunks=2,
        )
        enable_checkpointing(model)
        model.train()

        # A *pre*-hook is required here. Non-reentrant checkpointing aborts the
        # recompute as soon as it has the tensors it needs, so a regular forward
        # hook never fires for the interrupted block and hides the extra pass.
        counts: dict[int, int] = {}
        chunks = cast(Iterable[Iterable[nn.Module]], model.blocks)
        for idx, block in enumerate(m for chunk in chunks for m in chunk):
            if isinstance(block, nn.Identity):
                continue
            block.register_forward_pre_hook(
                lambda _m, _i, k=idx: counts.__setitem__(k, counts.get(k, 0) + 1)
            )

        x = torch.randn(2, 3, 56, 56, requires_grad=True)
        torch.autograd.backward(
            cast(DinoForwardFeatures, model).forward_features(x)[
                "x_norm_clstoken"
            ].sum()
        )

        assert counts, "no blocks were instrumented"
        # Exactly 2 = one forward plus one recompute. Checkpointing again inside
        # BlockChunk.forward makes the chunked blocks reach 3.
        assert max(counts.values()) == 2, counts
