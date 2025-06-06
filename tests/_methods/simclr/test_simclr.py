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

from lightly_train._methods.simclr.simclr import SimCLR, SimCLRArgs, SimCLRSGDArgs
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import ScalingInfo

from ...helpers import DummyCustomModel


class TestSimCLRArgs:
    def test_resolve_auto(self) -> None:
        args = SimCLRArgs()
        scaling_info = ScalingInfo(dataset_size=20_000, epochs=100)
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=AdamWArgs(),
            wrapped_model=DummyCustomModel(),
        )
        assert not args.has_auto()

    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            ("auto", SimCLRSGDArgs),
            (OptimizerType.ADAMW, AdamWArgs),
            (OptimizerType.SGD, SimCLRSGDArgs),
        ],
    )
    def test_optimizer_args_cls(
        self, optim_type: OptimizerType | Literal["auto"], expected: type[OptimizerArgs]
    ) -> None:
        assert SimCLR.optimizer_args_cls(optim_type=optim_type) == expected
