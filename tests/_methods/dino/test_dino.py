#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

import pytest

from lightly_train._methods.dino.dino import (
    DINO,
    DINOAdamWArgs,
    DINOArgs,
    DINOSGDArgs,
)
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
        assert args.warmup_teacher_temp_epochs == 30
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
        assert args.warmup_teacher_temp_epochs == 30
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
        assert args.warmup_teacher_temp_epochs == 3
        assert args.momentum_start == 0.996
        assert not args.has_auto()


class TestDINO:
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
