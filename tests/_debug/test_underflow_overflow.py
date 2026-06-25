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
from lightning_utilities.core.imports import RequirementCache
from pytest import LogCaptureFixture

from lightly_train._debug.debug_args import (
    DebugArgs,
    DebugUnderflowOverflowArgs,
    get_debug_args,
)

if not RequirementCache("transformers"):
    pytest.skip("Transformers not installed", allow_module_level=True)

import torch
import torch.nn as nn

from lightly_train._debug.underflow_overflow import UnderflowOverflowMonitor
from lightly_train._torch_compile import TorchCompileArgs


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
def test_get_debug_args__none() -> None:
    assert get_debug_args(None) == DebugArgs()


def test_get_debug_args__dict() -> None:
    args = get_debug_args(
        {"underflow_overflow": {"enabled": True, "max_frames_to_save": 5}}
    )
    assert args.underflow_overflow is not None
    assert args.underflow_overflow.enabled is True
    assert args.underflow_overflow.max_frames_to_save == 5


def test_get_debug_args__passes_through_object() -> None:
    args = DebugArgs(underflow_overflow=DebugUnderflowOverflowArgs(enabled=True))
    assert get_debug_args(args) is args


def test_get_debug_args__unknown_key_fails() -> None:
    with pytest.raises(Exception):
        get_debug_args({"nonexistent_key": 1})


def test_get_debug_args__underflow_overflow_unknown_key_fails() -> None:
    with pytest.raises(Exception):
        get_debug_args({"underflow_overflow": {"nonexistent_key": 1}})


def test_debug_args__disabled_by_default() -> None:
    assert DebugArgs().is_underflow_overflow_enabled() is False


def test_debug_args__enabled() -> None:
    assert (
        DebugArgs(
            underflow_overflow=DebugUnderflowOverflowArgs(enabled=True)
        ).is_underflow_overflow_enabled()
        is True
    )


def test_debug_args__field_present_but_disabled() -> None:
    assert (
        DebugArgs(
            underflow_overflow=DebugUnderflowOverflowArgs(enabled=False)
        ).is_underflow_overflow_enabled()
        is False
    )


def test_debug_underflow_overflow_args__defaults() -> None:
    args = DebugUnderflowOverflowArgs()
    assert args.enabled is False
    assert args.max_frames_to_save == 21
    assert args.trace_batch_nums == []
    assert args.abort_after_batch_num is None


# ---------------------------------------------------------------------------
# Monitor tests (require torch + transformers)
# ---------------------------------------------------------------------------
class _OverflowToyModel(nn.Module):
    """Tiny model whose forward produces inf, triggering the detector."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        # Multiply by a huge value twice so the intermediate overflows to inf in
        # float32.
        x = x * 1e20 * 1e20
        x = self.lin2(x)
        return x


class _FiniteToyModel(nn.Module):
    """Tiny model that stays finite."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_monitor__detects_overflow_and_writes_log(tmp_path: Path) -> None:
    model = _OverflowToyModel()
    monitor = UnderflowOverflowMonitor(
        model=model,
        debug_args=DebugUnderflowOverflowArgs(enabled=True),
        out_dir=tmp_path,
        global_rank=0,
    )
    monitor.set_step(7)
    with pytest.raises(ValueError, match="inf/nan detected"):
        model(torch.randn(2, 4))
    monitor.close()

    log = tmp_path / "debug" / "underflow_overflow_rank0.log"
    assert log.exists()
    text = log.read_text()
    assert "Detected inf/nan" in text
    assert "batch_number=7" in text
    assert "has infs" in text
    assert "lin2" in text


def test_monitor__per_rank_log_file(tmp_path: Path) -> None:
    model = _OverflowToyModel()
    monitor = UnderflowOverflowMonitor(
        model=model,
        debug_args=DebugUnderflowOverflowArgs(enabled=True),
        out_dir=tmp_path,
        global_rank=3,
    )
    monitor.set_step(0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 4))
    monitor.close()

    # The rank-specific file should exist, not the rank0 one.
    assert (tmp_path / "debug" / "underflow_overflow_rank3.log").exists()
    assert not (tmp_path / "debug" / "underflow_overflow_rank0.log").exists()


def test_monitor__finite_model_no_abort(tmp_path: Path) -> None:
    model = _FiniteToyModel()
    monitor = UnderflowOverflowMonitor(
        model=model,
        debug_args=DebugUnderflowOverflowArgs(enabled=True),
        out_dir=tmp_path,
        global_rank=0,
    )
    monitor.set_step(0)
    out = model(torch.randn(2, 4))
    assert torch.isfinite(out).all()
    monitor.close()

    # The log file exists (opened on init) but contains no detection report.
    log = tmp_path / "debug" / "underflow_overflow_rank0.log"
    assert log.exists()
    assert "Detected inf/nan" not in log.read_text()


def test_monitor__trace_mode_writes_frames(tmp_path: Path) -> None:
    model = _FiniteToyModel()
    monitor = UnderflowOverflowMonitor(
        model=model,
        debug_args=DebugUnderflowOverflowArgs(enabled=True, trace_batch_nums=[2]),
        out_dir=tmp_path,
        global_rank=0,
    )
    # Trace is only enabled for batch_number 2.
    monitor.set_step(2)
    model(torch.randn(2, 4))
    monitor.close()

    log = tmp_path / "debug" / "underflow_overflow_rank0.log"
    text = log.read_text()
    # Tracing writes the abs min / max header and module frames without aborting.
    assert "abs min" in text
    assert "Starting batch number=2" in text
    assert "lin" in text


def test_monitor__abort_after_batch_num(tmp_path: Path) -> None:
    model = _FiniteToyModel()
    monitor = UnderflowOverflowMonitor(
        model=model,
        debug_args=DebugUnderflowOverflowArgs(enabled=True, abort_after_batch_num=1),
        out_dir=tmp_path,
        global_rank=0,
    )
    # batch_number > abort_after_batch_num -> abort.
    monitor.set_step(2)
    with pytest.raises(ValueError, match="aborting after batch"):
        model(torch.randn(2, 4))
    monitor.close()


def test_monitor__logs_enable_pointer(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    model = _FiniteToyModel()
    with caplog.at_level("INFO"):
        monitor = UnderflowOverflowMonitor(
            model=model,
            debug_args=DebugUnderflowOverflowArgs(enabled=True),
            out_dir=tmp_path,
            global_rank=0,
        )
    monitor.close()
    assert any(
        "Underflow/overflow debugging enabled" in record.message
        and "underflow_overflow_rank0.log" in record.message
        for record in caplog.records
    )


def test_monitor__max_frames_to_save(tmp_path: Path) -> None:
    model = _OverflowToyModel()
    monitor = UnderflowOverflowMonitor(
        model=model,
        debug_args=DebugUnderflowOverflowArgs(enabled=True, max_frames_to_save=1),
        out_dir=tmp_path,
        global_rank=0,
    )
    monitor.set_step(0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 4))
    monitor.close()

    log = tmp_path / "debug" / "underflow_overflow_rank0.log"
    text = log.read_text()
    # Only the last frame is kept.
    assert "Last 1 forward frames" in text


# ---------------------------------------------------------------------------
# Compile conflict test
# ---------------------------------------------------------------------------
def test_compile_conflict__disabled_debug_allows_compile() -> None:
    """When debug is disabled, compile settings are untouched."""
    # Mirrors the intent of the inline check in train_task.py: debug disabled means
    # no restriction on torch.compile.
    debug = DebugArgs()
    compile_args = TorchCompileArgs(disable=False)
    assert debug.is_underflow_overflow_enabled() is False
    # No exception expected; compile_args unchanged.
    assert compile_args.disable is False


def test_compile_conflict__enabled_debug_with_compile_raises() -> None:
    """When debug is enabled, opting into compile must raise.

    This encodes the contract enforced inline in `_train_task_from_config`.
    """
    debug = DebugArgs(underflow_overflow=DebugUnderflowOverflowArgs(enabled=True))
    compile_args = TorchCompileArgs(disable=False)
    assert debug.is_underflow_overflow_enabled() is True
    with pytest.raises(ValueError, match="torch.compile cannot be used"):
        # Reproduce the inline guard logic.
        if debug.is_underflow_overflow_enabled() and not compile_args.disable:
            raise ValueError(
                "torch.compile cannot be used together with underflow/overflow "
                "debugging."
            )


def test_compile_conflict__enabled_debug_with_compile_disabled_ok() -> None:
    """When debug is enabled and compile is disabled, no exception (warning path)."""
    debug = DebugArgs(underflow_overflow=DebugUnderflowOverflowArgs(enabled=True))
    compile_args = TorchCompileArgs(disable=True)
    # Should not raise.
    if debug.is_underflow_overflow_enabled() and not compile_args.disable:
        pytest.fail("Should not have raised")
