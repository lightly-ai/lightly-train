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
import torch
from lightly.models.modules.memory_bank import MemoryBankModule

from lightly_train._methods.densecl.densecl import (
    DenseCL,
    DenseCLArgs,
    DenseCLSGDArgs,
    _AccumulationMemoryBank,
)
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import ScalingInfo

from ...helpers import DummyCustomModel


@pytest.mark.parametrize("sizes", [(4,), (2, 2), (1, 3)])
def test_memory_bank_accumulation(sizes: tuple[int, ...]) -> None:
    bank = MemoryBankModule(size=(8, 3))
    initial = bank.bank.clone()
    keys = torch.nn.functional.normalize(torch.arange(12).view(4, 3).float(), dim=-1)
    staged = _AccumulationMemoryBank(bank)
    for part in keys.split(sizes):
        _, negatives = staged(part, update=True)
        torch.testing.assert_close(negatives, initial.T)
        torch.testing.assert_close(staged.bank, initial)
        assert staged.bank_ptr.item() == 0
    staged.commit()
    torch.testing.assert_close(staged.bank[:4], keys)
    torch.testing.assert_close(staged.bank[4:], initial[4:])
    assert staged.bank_ptr.item() == 4
    staged.commit()
    assert staged.bank_ptr.item() == 4


class TestDenseCLArgs:
    def test_resolve_auto(self) -> None:
        args = DenseCLArgs()
        scaling_info = ScalingInfo(dataset_size=20_000, epochs=100)
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=AdamWArgs(),
            wrapped_model=DummyCustomModel(),
        )
        assert args.memory_bank_size == 8192
        assert not args.has_auto()

    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            ("auto", DenseCLSGDArgs),
            (OptimizerType.ADAMW, AdamWArgs),
            (OptimizerType.SGD, DenseCLSGDArgs),
        ],
    )
    def test_optimizer_args_cls(
        self, optim_type: OptimizerType | Literal["auto"], expected: type[OptimizerArgs]
    ) -> None:
        assert DenseCL.optimizer_args_cls(optim_type=optim_type) == expected
