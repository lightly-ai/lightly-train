#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from lightning_utilities.core.imports import RequirementCache

from lightly_train._debug.debug_args import (
    DebugArgs,
    DebugUnderflowOverflowArgs,
    get_debug_args,
)
from lightly_train.errors import ConfigValidationError

if not RequirementCache("transformers"):
    pytest.skip("Transformers not installed", allow_module_level=True)

import torch
import torch.nn as nn

from lightly_train._debug.underflow_overflow import (
    UnderflowOverflowMonitor,
    check_compile_conflict,
)
from lightly_train._torch_compile import TorchCompileArgs


class _OverflowToyModel(nn.Module):
    """Tiny model whose forward produces inf, triggering the detector."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = x * 1e20 * 1e20
        x = self.lin2(x)
        return x


class _FiniteToyModel(nn.Module):
    """Tiny model that stays finite."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)


@pytest.fixture
def make_monitor(
    tmp_path: Path,
) -> Iterator[Callable[..., UnderflowOverflowMonitor]]:
    created: list[UnderflowOverflowMonitor] = []

    def _make(
        *,
        model: nn.Module | None = None,
        debug_args: DebugUnderflowOverflowArgs | None = None,
        global_rank: int = 0,
    ) -> UnderflowOverflowMonitor:
        monitor = UnderflowOverflowMonitor(
            model=model if model is not None else _FiniteToyModel(),
            debug_args=debug_args or DebugUnderflowOverflowArgs(enabled=True),
            out_dir=tmp_path,
            global_rank=global_rank,
        )
        created.append(monitor)
        return monitor

    yield _make

    for monitor in created:
        monitor.close()


def test_get_debug_args__parses_user_config() -> None:
    assert get_debug_args(None).is_underflow_overflow_enabled() is False

    args = get_debug_args(
        {"underflow_overflow": {"enabled": True, "max_frames_to_save": 5}}
    )

    assert args.is_underflow_overflow_enabled() is True
    assert args.underflow_overflow is not None
    assert args.underflow_overflow.max_frames_to_save == 5


def test_get_debug_args__rejects_unknown_keys() -> None:
    with pytest.raises(ConfigValidationError):
        get_debug_args({"underflow_overflow": {"nonexistent_key": 1}})


def test_check_compile_conflict__disabled_debug_allows_compile() -> None:
    check_compile_conflict(DebugArgs(), TorchCompileArgs(disable=False))


def test_check_compile_conflict__enabled_debug_with_compile_raises() -> None:
    debug = DebugArgs(underflow_overflow=DebugUnderflowOverflowArgs(enabled=True))

    with pytest.raises(ValueError, match="torch.compile cannot be used"):
        check_compile_conflict(debug, TorchCompileArgs(disable=False))


class TestUnderflowOverflowMonitor:
    def test_detects_overflow_and_writes_log(
        self,
        tmp_path: Path,
        make_monitor: Callable[..., UnderflowOverflowMonitor],
    ) -> None:
        model = _OverflowToyModel()
        monitor = make_monitor(model=model, global_rank=3)
        monitor.set_step(7)

        with pytest.raises(ValueError, match="inf/nan detected"):
            model(torch.randn(2, 4))

        log = tmp_path / "debug" / "underflow_overflow_rank3.log"
        assert log.exists()
        text = log.read_text()
        assert "Detected inf/nan" in text
        assert "batch_number=7" in text
        assert "lin2" in text

    def test_finite_model_does_not_abort(
        self,
        make_monitor: Callable[..., UnderflowOverflowMonitor],
    ) -> None:
        model = _FiniteToyModel()
        monitor = make_monitor(model=model)
        monitor.set_step(0)

        out = model(torch.randn(2, 4))

        assert torch.isfinite(out).all()

    def test_trace_mode_writes_report_without_aborting(
        self,
        tmp_path: Path,
        make_monitor: Callable[..., UnderflowOverflowMonitor],
    ) -> None:
        model = _FiniteToyModel()
        monitor = make_monitor(
            model=model,
            debug_args=DebugUnderflowOverflowArgs(enabled=True, trace_batch_nums=[2]),
        )
        monitor.set_step(2)

        model(torch.randn(2, 4))

        text = (tmp_path / "debug" / "underflow_overflow_rank0.log").read_text()
        assert "abs min" in text
        assert "lin" in text

    def test_abort_after_batch_num(
        self,
        make_monitor: Callable[..., UnderflowOverflowMonitor],
    ) -> None:
        model = _FiniteToyModel()
        monitor = make_monitor(
            model=model,
            debug_args=DebugUnderflowOverflowArgs(enabled=True, abort_after_batch_num=1),
        )
        monitor.set_step(2)

        with pytest.raises(ValueError, match="aborting after batch"):
            model(torch.randn(2, 4))

    def test_close__allows_later_forwards(
        self,
        make_monitor: Callable[..., UnderflowOverflowMonitor],
    ) -> None:
        model = _FiniteToyModel()
        monitor = make_monitor(model=model)
        monitor.close()

        model(torch.randn(2, 4))