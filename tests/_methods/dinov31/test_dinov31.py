#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Size, Tensor
from torch.nn import Module

from lightly_train._methods.dinov2.dinov2 import DINOv2, DINOv2AdamWViTArgs, DINOv2Args
from lightly_train._methods.dinov31.dinov31 import (
    DINOv31,
    DINOv31AdamW8bitArgs,
    DINOv31Args,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import ScalingInfo
from lightly_train.types import Batch

from ...helpers import dummy_dinov2_vit_model


def setup_dinov31_helper(
    dinov31_args: DINOv31Args,
    mocker: MockerFixture,
    emb_model: EmbeddingModel,
    batch_size: int,
) -> DINOv31:
    optimizer_args = DINOv2AdamWViTArgs()
    scaling_info = ScalingInfo(dataset_size=1000, epochs=100)
    dinov31_args.resolve_auto(
        scaling_info=scaling_info,
        optimizer_args=optimizer_args,
        wrapped_model=emb_model.wrapped_model,
    )
    dinov31 = DINOv31(
        method_args=dinov31_args,
        optimizer_args=optimizer_args,
        embedding_model=emb_model,
        global_batch_size=batch_size,
        num_input_channels=3,
    )
    trainer_mock = mocker.Mock()
    trainer_mock.global_step = 0
    trainer_mock.max_epochs = 1
    trainer_mock.estimated_stepping_batches = 1
    dinov31.trainer = trainer_mock
    return dinov31


def _full_geom(batch_size: int, image_size: int = 8) -> Tensor:
    # [x0, y0, x1, y1, image_w, image_h, hflip, vflip] = full image, no flip.
    g = torch.tensor(
        [0.0, 0.0, image_size, image_size, image_size, image_size, 0.0, 0.0],
        dtype=torch.float32,
    )
    return g.repeat(batch_size, 1)


def _sub_geom(batch_size: int, image_size: int = 8) -> Tensor:
    # A small box inside the full image so it overlaps the global crop.
    g = torch.tensor(
        [2.0, 2.0, 6.0, 6.0, image_size, image_size, 0.0, 0.0],
        dtype=torch.float32,
    )
    return g.repeat(batch_size, 1)


def _dinov31_batch(
    batch_size: int,
    n_dino_local: int = 2,
    n_paka_local: int = 8,
) -> Batch:
    """Builds the DINOv31 view layout + geometries (patch_size=2 dummy model)."""
    views = (
        [torch.rand(batch_size, 3, 8, 8) for _ in range(2)]  # globals
        + [torch.rand(batch_size, 3, 4, 4) for _ in range(n_dino_local)]  # dino locals
        + [torch.rand(batch_size, 3, 8, 8) for _ in range(2)]  # clean globals
        + [torch.rand(batch_size, 3, 4, 4) for _ in range(n_paka_local)]  # paka locals
    )
    geometries = (
        [_full_geom(batch_size) for _ in range(2)]
        + [_full_geom(batch_size) for _ in range(n_dino_local)]
        + [_full_geom(batch_size) for _ in range(2)]
        + [_sub_geom(batch_size) for _ in range(n_paka_local)]
    )
    return {
        "views": views,
        "geometries": geometries,
        "filename": [f"img_{i}" for i in range(batch_size)],
    }


class TestDINOv31:
    def test_train_step_impl(self, mocker: MockerFixture) -> None:
        b = 4
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        batch = _dinov31_batch(b)
        dinov31 = setup_dinov31_helper(DINOv31Args(), mocker, emb_model, b)

        out = dinov31.training_step_impl(batch, 0)

        assert out.loss.shape == Size([])
        assert torch.isfinite(out.loss)
        assert out.log_dict is not None
        assert "train_loss/paka_loss" in out.log_dict
        assert torch.isfinite(out.log_dict["train_loss/paka_loss"])
        for key in ("dino_global_loss", "dino_local_loss", "ibot_loss", "koleo_loss"):
            assert torch.isfinite(out.log_dict[f"train_loss/{key}"])

    def test_paka_skipped_before_start_step(self, mocker: MockerFixture) -> None:
        # With paka_start_step in the future, PaKA is skipped and no paka loss is
        # logged (the step is a plain DINOv2 step on the leading views).
        b = 4
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        batch = _dinov31_batch(b)
        dinov31 = setup_dinov31_helper(
            DINOv31Args(paka_start_step=10), mocker, emb_model, b
        )

        out = dinov31.training_step_impl(batch, 0)

        assert out.log_dict is not None
        assert "train_loss/paka_loss" not in out.log_dict
        assert torch.isfinite(out.loss)

    def test_dino_path_unchanged_by_paka(self, mocker: MockerFixture) -> None:
        # DINOv31 calls the inherited DINOv2 step on ONLY the leading views
        # (super().training_step_impl), then adds PaKA. So the four DINO
        # sub-losses must equal a direct DINOv2 step on those same views, on the
        # SAME instance (same weights/heads) with the same seed => same iBOT
        # masks. This verifies the view split does not perturb the DINO path.
        b = 4
        n_dino_local = 2
        batch = _dinov31_batch(b, n_dino_local=n_dino_local)
        n_dino = 2 + n_dino_local
        dino_batch: Batch = {
            "views": batch["views"][:n_dino],
            "filename": batch["filename"],
        }

        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov31 = setup_dinov31_helper(DINOv31Args(), mocker, emb_model, b)
        # eval() freezes BN running stats (no drift across the two calls); the
        # centers are reset before each call. Together this makes the two steps
        # bit-identical so the comparison isolates the view split.
        dinov31.eval()

        def reset_centers() -> None:
            # Zero the center AND drop any pending async center update: the
            # first step ends with update_center() leaving updated=False plus a
            # pending batch center, which the second step would otherwise apply
            # before centering (making its center differ from the first step's).
            for loss in (dinov31.dino_loss, dinov31.ibot_loss):
                center = getattr(loss, "center", None)
                if isinstance(center, Tensor):
                    with torch.no_grad():
                        center.zero_()
                loss.updated = True
                loss.reduce_handle = None

        # The iBOT MaskingGenerator draws from Python's `random` module (not
        # torch), so both RNGs must be seeded for identical masks across the
        # two calls.
        reset_centers()
        torch.manual_seed(123)
        random.seed(123)
        out_v31 = dinov31.training_step_impl(batch, 0)  # super(dino_views) + PaKA
        reset_centers()
        torch.manual_seed(123)
        random.seed(123)
        out_dino = DINOv2.training_step_impl(
            dinov31, dino_batch, 0
        )  # plain DINOv2 step

        assert out_v31.log_dict is not None and out_dino.log_dict is not None
        # The DINO sub-losses match the direct DINOv2 step: super() receives only
        # the leading views, so the DINO path is unchanged by PaKA. dino_local_loss
        # is the one that exercises the local-view split. (koleo_loss is excluded:
        # at b=4 its -log(min-distance) is dominated by noise, and it only uses
        # globals so it does not test the split anyway.) With centers fully reset
        # the two steps are bit-identical, so the tolerance only absorbs float
        # noise; a real view-split bug would change these by orders of magnitude.
        for key in ("dino_global_loss", "dino_local_loss", "ibot_loss"):
            assert torch.allclose(
                out_v31.log_dict[f"train_loss/{key}"],
                out_dino.log_dict[f"train_loss/{key}"],
                atol=1e-5,
            )

    def test_paka_head_excluded_from_export(self, mocker: MockerFixture) -> None:
        # The PaKA head lives on the Method, not on the embedding/wrapped model,
        # so export (which serializes wrapped_model/embedding_model) excludes it.
        b = 4
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov31 = setup_dinov31_helper(DINOv31Args(), mocker, emb_model, b)

        assert isinstance(dinov31.student_paka_head, Module)
        # Absent from the wrapped model that export serializes...
        wrapped_keys = set(
            dinov31.teacher_embedding_model.wrapped_model.state_dict().keys()
        )
        assert not any("paka_head" in k for k in wrapped_keys)
        # ...but present on the full Method state (it is a Method attribute).
        method_keys = set(dinov31.state_dict().keys())
        assert any("student_paka_head" in k for k in method_keys)
        assert any("teacher_paka_head" in k for k in method_keys)
        # Student head trainable + in the optimizer; teacher head frozen.
        assert all(p.requires_grad for p in dinov31.student_paka_head.parameters())
        assert not any(p.requires_grad for p in dinov31.teacher_paka_head.parameters())
        assert dinov31.student_paka_head in dinov31.trainable_modules().modules

    def test_lenient_checkpoint_load(self, mocker: MockerFixture) -> None:
        # A DINOv2 state dict (no paka_head keys) loads into DINOv31 without
        # error, while a real mismatch still raises.
        b = 4
        emb_model = EmbeddingModel(wrapped_model=dummy_dinov2_vit_model())
        dinov31 = setup_dinov31_helper(DINOv31Args(), mocker, emb_model, b)

        # Plain DINOv2 state dict (no paka_head keys).
        from tests._methods.dinov2.test_dinov2 import setup_dinov2_helper

        dinov2 = setup_dinov2_helper(DINOv2Args(), mocker, emb_model, b)
        dinov2_sd = dinov2.state_dict()
        assert not any("paka_head" in k for k in dinov2_sd)
        # Must not raise (paka_head keys tolerated as missing).
        dinov31.load_state_dict(dinov2_sd, strict=True)

        # A genuine unexpected key still raises.
        bad_sd = dict(dinov2_sd)
        bad_sd["totally.unexpected.key"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            dinov31.load_state_dict(bad_sd, strict=True)

    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            (OptimizerType.ADAMW8BIT, DINOv31AdamW8bitArgs),
            ("auto", DINOv2AdamWViTArgs),
            (OptimizerType.ADAMW, DINOv2AdamWViTArgs),
        ],
    )
    def test_optimizer_args_cls(self, optim_type: Any, expected: type) -> None:
        # 8-bit AdamW resolves to the DINOv31 args; everything else delegates to
        # DINOv2 unchanged.
        assert DINOv31.optimizer_args_cls(optim_type=optim_type) == expected


class TestDINOv31Args:
    def test_defaults(self) -> None:
        args = DINOv31Args()
        assert args.paka_weight == 1.0
        assert args.paka_start_step == 0
        assert args.paka_num_local == 8
        assert args.paka_max_tokens == 512
