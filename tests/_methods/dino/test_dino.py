#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset, default_collate

from lightly_train._methods.dino.dino import (
    DINO,
    DINOAdamWArgs,
    DINOArgs,
    DINOSGDArgs,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo

from ...helpers import DummyCustomModel


class TestDINOArgs:
    def test_resolve_auto__default_scaling_info(self) -> None:
        args = DINOArgs()
        scaling_info = ScalingInfo(dataset_size=IMAGENET_SIZE, epochs=100)
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=DINOAdamWArgs(),
            wrapped_model=DummyCustomModel(),
        )
        assert args.output_dim == 65536
        assert args.teacher_temp == 0.07
        assert args.warmup_teacher_temp == 0.04
        assert args.warmup_teacher_temp_epochs is None
        assert args.warmup_teacher_temp_steps == 37500
        assert args.student_freeze_last_layer_epochs is None
        assert args.student_freeze_last_layer_steps == 1250
        assert args.momentum_start == 0.996
        assert not args.has_auto()

    def test_resolve_auto__lower_dataset_size(self) -> None:
        args = DINOArgs()
        scaling_info = ScalingInfo(dataset_size=20_000, epochs=100)
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=DINOAdamWArgs(),
            wrapped_model=DummyCustomModel(),
        )
        assert args.output_dim == 2048
        assert args.teacher_temp == 0.02
        assert args.warmup_teacher_temp == 0.02
        assert args.warmup_teacher_temp_epochs is None
        assert args.warmup_teacher_temp_steps == 37500
        assert args.student_freeze_last_layer_epochs is None
        assert args.student_freeze_last_layer_steps == 1250
        assert args.momentum_start == 0.99
        assert not args.has_auto()

    def test_resolve_auto__fewer_epochs(self) -> None:
        args = DINOArgs()
        scaling_info = ScalingInfo(dataset_size=IMAGENET_SIZE, epochs=10)
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=DINOAdamWArgs(),
            wrapped_model=DummyCustomModel(),
        )
        assert args.output_dim == 65536
        assert args.teacher_temp == 0.07
        assert args.warmup_teacher_temp == 0.04
        assert args.warmup_teacher_temp_epochs is None
        assert args.warmup_teacher_temp_steps == 37500
        assert args.student_freeze_last_layer_epochs is None
        assert args.student_freeze_last_layer_steps == 1250
        assert args.momentum_start == 0.996
        assert not args.has_auto()


class TestDINO:
    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    def test_center_accumulation_matches_full_batch(self, dtype: torch.dtype) -> None:
        from lightly.loss import DINOLoss

        from lightly_train._methods.dino.dino import _AccumulatingDINOLoss

        reference = DINOLoss(output_dim=4, center_momentum=0.9)
        accumulated = _AccumulatingDINOLoss(output_dim=4, center_momentum=0.9)
        # A half-precision sum over the whole window would overflow, while its mean is finite.
        teacher = torch.full((2, 64, 4), 1000.0, dtype=dtype)
        for start in range(0, 64, 16):
            accumulated.update_center(teacher[:, start : start + 16])
            assert torch.count_nonzero(accumulated.center) == 0
        accumulated.commit_center()
        reference.update_center(teacher)
        torch.testing.assert_close(accumulated.center, reference.center, rtol=0, atol=0)
        accumulated.commit_center()
        torch.testing.assert_close(accumulated.center, reference.center, rtol=0, atol=0)

    @pytest.mark.parametrize("num_examples", [12, 10])
    def test_gradient_accumulation_preserves_teacher_and_center(
        self, tmp_path: Path, num_examples: int
    ) -> None:
        def run(batch_size: int, accumulation: int) -> dict[str, torch.Tensor]:
            torch.manual_seed(19)
            method = DINO(
                method_args=DINOArgs(
                    output_dim=8,
                    hidden_dim=16,
                    bottleneck_dim=4,
                    teacher_temp=0.07,
                    warmup_teacher_temp=0.07,
                    warmup_teacher_temp_steps=0,
                    student_freeze_last_layer_steps=0,
                    momentum_start=0.9,
                    momentum_end=0.9,
                    weight_decay_start=0.0,
                    weight_decay_end=0.0,
                    warmup_steps=0,
                    reference_batch_size=4,
                ),
                optimizer_args=DINOSGDArgs(lr=0.01, momentum=0.0, weight_decay=0.0),
                embedding_model=EmbeddingModel(DummyCustomModel(feature_dim=4)),
                global_batch_size=4,
                num_input_channels=3,
            )
            # A resumed teacher can differ from the student before the first window.
            with torch.no_grad():
                for parameter in method.teacher_embedding_model.parameters():
                    parameter.add_(0.1)
            images = torch.randn(num_examples, 3, 4, 4)
            data = TensorDataset(images, images * 0.5 + 0.1)
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_epochs=1,
                accumulate_grad_batches=accumulation,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                default_root_dir=tmp_path,
            )
            trainer.fit(
                method,
                train_dataloaders=DataLoader(
                    data,
                    batch_size=batch_size,
                    collate_fn=lambda views: {"views": default_collate(views)},
                ),
            )
            assert trainer.global_step == (num_examples + 3) // 4
            return {
                name: value.detach().clone()
                for name, value in method.state_dict().items()
            }

        full = run(batch_size=4, accumulation=1)
        accumulated = run(batch_size=1, accumulation=4)
        for name, expected in full.items():
            # Lightning separately controls loss normalization in partial windows.
            # The last partial window must still commit the correct teacher and center.
            if num_examples % 4 and not (
                "teacher" in name or name == "criterion.center"
            ):
                continue
            torch.testing.assert_close(
                accumulated[name], expected, rtol=1e-5, atol=1e-6, msg=name
            )

    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            ("auto", DINOSGDArgs),
            (OptimizerType.ADAMW, DINOAdamWArgs),
            (OptimizerType.SGD, DINOSGDArgs),
        ],
    )
    def test_optimizer_args_cls(
        self, optim_type: OptimizerType | Literal["auto"], expected: type[OptimizerArgs]
    ) -> None:
        assert DINO.optimizer_args_cls(optim_type=optim_type) == expected
