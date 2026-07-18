#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pytest_mock import MockerFixture

from lightly_train._torch_compile import TorchCompileArgs, try_compile


def test_try_compile__runtime_failure_falls_back_once(mocker: MockerFixture) -> None:
    compiled_mock = mocker.Mock(side_effect=RuntimeError("inductor failed"))
    mocker.patch("torch.compile", return_value=compiled_mock)

    def eager(x: int) -> int:
        return x + 1

    fn = try_compile(eager, "test_fn", TorchCompileArgs(disable=False, mode="default"))

    assert fn(1) == 2
    assert fn(2) == 3
    compiled_mock.assert_called_once_with(1)


def test_try_compile__passes_compile_args(mocker: MockerFixture) -> None:
    def eager(x: int) -> int:
        return x + 1

    def compiled(x: int) -> int:
        return x + 2

    compile_mock = mocker.patch("torch.compile", return_value=compiled)

    fn = try_compile(eager, "test_fn", TorchCompileArgs(disable=False, mode="default"))

    assert fn(1) == 3
    compile_mock.assert_called_once()
    assert compile_mock.call_args.args[0] is eager
    assert compile_mock.call_args.kwargs["mode"] == "default"
