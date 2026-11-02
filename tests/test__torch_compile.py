#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

import pytest
from pytest_mock import MockerFixture
from torch._dynamo.exc import TorchDynamoException

from lightly_train._torch_compile import TorchCompileArgs, try_compile


def test_try_compile__compile_failure_falls_back_once(mocker: MockerFixture) -> None:
    # Force a non-Windows platform as try_compile skips compilation on Windows.
    mocker.patch("sys.platform", "linux")
    compiled_mock = mocker.Mock(side_effect=TorchDynamoException("inductor failed"))
    mocker.patch("torch.compile", return_value=compiled_mock)

    def eager(x: int) -> int:
        return x + 1

    fn = try_compile(eager, "test_fn", TorchCompileArgs(disable=False, mode="default"))

    assert fn(1) == 2
    assert fn(2) == 3
    compiled_mock.assert_called_once_with(1)


def test_try_compile__non_compile_error_propagates(mocker: MockerFixture) -> None:
    # Force a non-Windows platform as try_compile skips compilation on Windows.
    mocker.patch("sys.platform", "linux")
    compiled_mock = mocker.Mock(side_effect=[RuntimeError("out of memory"), 3])
    mocker.patch("torch.compile", return_value=compiled_mock)

    def eager(x: int) -> int:
        return x + 1

    fn = try_compile(eager, "test_fn", TorchCompileArgs(disable=False))

    with pytest.raises(RuntimeError, match="out of memory"):
        fn(1)
    assert fn(1) == 3


def test_try_compile__default_args(mocker: MockerFixture) -> None:
    # Force a non-Windows platform as try_compile skips compilation on Windows.
    mocker.patch("sys.platform", "linux")
    compile_mock = mocker.patch("torch.compile")

    def eager(x: int) -> int:
        return x + 1

    try_compile(eager, "test_fn", TorchCompileArgs(disable=False))

    assert compile_mock.call_args.kwargs["mode"] is None
    assert compile_mock.call_args.kwargs["options"] is None


def test_try_compile__options_without_mode(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    # Force a non-Windows platform as try_compile skips compilation on Windows.
    mocker.patch("sys.platform", "linux")

    def eager(x: int) -> int:
        return x + 1

    with caplog.at_level(logging.WARNING, logger="lightly_train._torch_compile"):
        try_compile(eager, "test_fn", TorchCompileArgs(disable=False, options={}))

    assert "Compilation failed" not in caplog.text


def test_try_compile__passes_compile_args(mocker: MockerFixture) -> None:
    # Force a non-Windows platform as try_compile skips compilation on Windows.
    mocker.patch("sys.platform", "linux")

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


def test_try_compile__windows_skips_compilation(mocker: MockerFixture) -> None:
    mocker.patch("sys.platform", "win32")
    compile_mock = mocker.patch("torch.compile")

    def eager(x: int) -> int:
        return x + 1

    fn = try_compile(eager, "test_fn", TorchCompileArgs(disable=False, mode="default"))

    compile_mock.assert_not_called()
    assert fn(1) == 2
