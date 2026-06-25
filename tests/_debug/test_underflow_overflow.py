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
from pydantic import ValidationError
from pytest import LogCaptureFixture

from lightly_train._debug.debug_args import (
    DebugArgs,
    DebugUnderflowOverflowArgs,
    get_debug_args,
)
from lightly_train.errors import ConfigValidationError

if not RequirementCache("transformers"):
    pytest.skip("Transformers not installed", allow_module_level=True)

import sys

import torch
import torch.nn as nn

from lightly_train._debug.underflow_overflow import (
    UnderflowOverflowMonitor,
    check_compile_conflict,
)
from lightly_train._torch_compile import TorchCompileArgs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def make_monitor(tmp_path: Path):
    """Factory fixture that creates monitors and closes them on teardown."""
    created: list[UnderflowOverflowMonitor] = []

    def _make(
        debug_args: DebugUnderflowOverflowArgs | None = None,
        model: nn.Module | None = None,
        global_rank: int = 0,
        out_dir: Path | None = None,
    ) -> UnderflowOverflowMonitor:
        monitor = UnderflowOverflowMonitor(
            model=model if model is not None else _FiniteToyModel(),
            debug_args=debug_args or DebugUnderflowOverflowArgs(enabled=True),
            out_dir=out_dir if out_dir is not None else tmp_path,
            global_rank=global_rank,
        )
        created.append(monitor)
        return monitor

    yield _make
    for monitor in created:
        monitor.close()


# ---------------------------------------------------------------------------
# Toy models
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


class _NanToyModel(nn.Module):
    """Tiny model whose forward produces NaN (0 * inf)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * float("inf") * 0.0


class _FiniteToyModel(nn.Module):
    """Tiny model that stays finite."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


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
    with pytest.raises(ConfigValidationError):
        get_debug_args({"nonexistent_key": 1})


def test_get_debug_args__underflow_overflow_unknown_key_fails() -> None:
    with pytest.raises(ConfigValidationError):
        get_debug_args({"underflow_overflow": {"nonexistent_key": 1}})


def test_get_debug_args__explicit_enabled_false_dict() -> None:
    args = get_debug_args({"underflow_overflow": {"enabled": False}})
    assert args.is_underflow_overflow_enabled() is False
    assert args.underflow_overflow is not None
    assert args.underflow_overflow.enabled is False


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


def test_debug_args__enabled_false_with_other_fields() -> None:
    """enabled=False is a hard override regardless of trace/abort config."""
    args = DebugArgs(
        underflow_overflow=DebugUnderflowOverflowArgs(
            enabled=False,
            trace_batch_nums=[1, 2],
            abort_after_batch_num=5,
        )
    )
    assert args.is_underflow_overflow_enabled() is False


def test_debug_underflow_overflow_args__defaults() -> None:
    args = DebugUnderflowOverflowArgs()
    assert args.enabled is False
    assert args.max_frames_to_save == 21
    assert args.trace_batch_nums == []
    assert args.abort_after_batch_num is None


def test_debug_underflow_overflow_args__negative_max_frames_rejected() -> None:
    with pytest.raises(ValidationError):
        DebugUnderflowOverflowArgs(max_frames_to_save=0)
    with pytest.raises(ValidationError):
        DebugUnderflowOverflowArgs(max_frames_to_save=-1)


def test_debug_underflow_overflow_args__negative_abort_rejected() -> None:
    with pytest.raises(ValidationError):
        DebugUnderflowOverflowArgs(abort_after_batch_num=-1)


def test_debug_underflow_overflow_args__negative_trace_batch_nums_rejected() -> None:
    with pytest.raises(ValidationError):
        DebugUnderflowOverflowArgs(trace_batch_nums=[1, -2])


def test_debug_underflow_overflow_args__large_abort_ok() -> None:
    args = DebugUnderflowOverflowArgs(abort_after_batch_num=10**12)
    assert args.abort_after_batch_num == 10**12


# ---------------------------------------------------------------------------
# Monitor tests (require torch + transformers)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)


def test_monitor__subclass_is_lightly(tmp_path: Path) -> None:
    """The monitor wraps a LightlyTrain subclass of DebugUnderflowOverflow."""
    monitor = UnderflowOverflowMonitor(
        model=_FiniteToyModel(),
        debug_args=DebugUnderflowOverflowArgs(enabled=True),
        out_dir=tmp_path,
        global_rank=0,
    )
    try:
        from transformers.debug_utils import DebugUnderflowOverflow

        assert isinstance(monitor._hf, DebugUnderflowOverflow)
        assert type(monitor._hf).__name__ == "_LightlyDebugUnderflowOverflow"
    finally:
        monitor.close()


def test_monitor__creates_out_dir_if_missing(tmp_path: Path) -> None:
    """out_dir is created with parents=True (load-bearing for callers)."""
    nested = tmp_path / "deeply" / "nested" / "out"
    monitor = UnderflowOverflowMonitor(
        model=_FiniteToyModel(),
        debug_args=DebugUnderflowOverflowArgs(enabled=True),
        out_dir=nested,
        global_rank=0,
    )
    monitor.close()
    assert nested.exists()
    assert (nested / "debug" / "underflow_overflow_rank0.log").exists()


def test_monitor__detects_overflow_and_writes_log(tmp_path: Path, make_monitor) -> None:
    model = _OverflowToyModel()
    monitor = make_monitor(model=model)
    monitor.set_step(7)
    with pytest.raises(ValueError, match="inf/nan detected"):
        model(torch.randn(2, 4))

    log = tmp_path / "debug" / "underflow_overflow_rank0.log"
    assert log.exists()
    text = log.read_text()
    assert "Detected inf/nan" in text
    assert "batch_number=7" in text
    assert "has infs" in text
    assert "lin2" in text


def test_monitor__detects_nan(tmp_path: Path, make_monitor) -> None:
    model = _NanToyModel()
    monitor = make_monitor(model=model)
    monitor.set_step(0)
    with pytest.raises(ValueError, match="inf/nan detected"):
        model(torch.randn(2, 4))

    text = (tmp_path / "debug" / "underflow_overflow_rank0.log").read_text()
    assert "Detected inf/nan" in text
    assert "has nans" in text


def test_monitor__per_rank_log_file(tmp_path: Path, make_monitor) -> None:
    model = _OverflowToyModel()
    monitor = make_monitor(model=model, global_rank=3)
    monitor.set_step(0)
    with pytest.raises(ValueError):
        model(torch.randn(2, 4))

    assert (tmp_path / "debug" / "underflow_overflow_rank3.log").exists()
    assert not (tmp_path / "debug" / "underflow_overflow_rank0.log").exists()


def test_monitor__finite_model_no_abort(tmp_path: Path, make_monitor) -> None:
    monitor = make_monitor()  # default = _FiniteToyModel
    monitor.set_step(0)
    out = monitor._hf.model(torch.randn(2, 4))
    assert torch.isfinite(out).all()

    log = tmp_path / "debug" / "underflow_overflow_rank0.log"
    assert log.exists()
    text = log.read_text()
    # No overflow dump is written for finite forward passes.
    assert "Detected inf/nan" not in text


def test_monitor__trace_mode_writes_frames(tmp_path: Path, make_monitor) -> None:
    monitor = make_monitor(
        debug_args=DebugUnderflowOverflowArgs(enabled=True, trace_batch_nums=[2])
    )
    monitor.set_step(2)
    monitor._hf.model(torch.randn(2, 4))

    text = (tmp_path / "debug" / "underflow_overflow_rank0.log").read_text()
    # Tracing writes the abs min / max header and module frames without aborting.
    assert "abs min" in text
    assert "lin" in text


def test_monitor__abort_after_batch_num(tmp_path: Path, make_monitor) -> None:
    monitor = make_monitor(
        debug_args=DebugUnderflowOverflowArgs(enabled=True, abort_after_batch_num=1)
    )
    # batch_number > abort_after_batch_num -> abort.
    monitor.set_step(2)
    with pytest.raises(ValueError, match="aborting after batch"):
        monitor._hf.model(torch.randn(2, 4))


def test_monitor__logs_enable_pointer(
    tmp_path: Path, caplog: LogCaptureFixture, make_monitor
) -> None:
    with caplog.at_level("INFO"):
        make_monitor()
    assert any(
        "Underflow/overflow debugging enabled" in record.message
        and "underflow_overflow_rank0.log" in record.message
        for record in caplog.records
    )


def test_monitor__max_frames_to_save(tmp_path: Path, make_monitor) -> None:
    monitor = make_monitor(
        model=_OverflowToyModel(),
        debug_args=DebugUnderflowOverflowArgs(enabled=True, max_frames_to_save=1),
    )
    monitor.set_step(0)
    with pytest.raises(ValueError):
        monitor._hf.model(torch.randn(2, 4))

    text = (tmp_path / "debug" / "underflow_overflow_rank0.log").read_text()
    # Only the last frame is kept.
    assert "Last 1 forward frames" in text


def test_monitor__set_step_advances_batch_number_each_call(
    tmp_path: Path, make_monitor
) -> None:
    """set_step drives the batch number that ends up in the overflow dump.

    Each forward uses the overflow model so the dump includes the batch_number,
    which lets us assert that the monitor correctly aligned with the outer
    LightlyTrain step counter at each call.
    """
    monitor = make_monitor(model=_OverflowToyModel())
    for step in (3, 4, 5):
        monitor.set_step(step)
        with pytest.raises(ValueError, match="inf/nan detected"):
            monitor._hf.model(torch.randn(2, 4))

    text = (tmp_path / "debug" / "underflow_overflow_rank0.log").read_text()
    assert "Detected inf/nan during batch_number=3" in text
    assert "Detected inf/nan during batch_number=4" in text
    assert "Detected inf/nan during batch_number=5" in text


def test_monitor__set_step_backwards_re_emits(tmp_path: Path, make_monitor) -> None:
    """Going backwards (10 -> 3) still produces a fresh dump with the new number."""
    monitor = make_monitor(model=_OverflowToyModel())
    monitor.set_step(10)
    with pytest.raises(ValueError):
        monitor._hf.model(torch.randn(2, 4))
    monitor.set_step(3)  # backwards
    with pytest.raises(ValueError):
        monitor._hf.model(torch.randn(2, 4))

    text = (tmp_path / "debug" / "underflow_overflow_rank0.log").read_text()
    assert text.count("Detected inf/nan during batch_number=") == 2
    assert "batch_number=10" in text
    assert "batch_number=3" in text


def test_monitor__redirect_stdout_reverts_after_forward(
    tmp_path: Path, make_monitor
) -> None:
    """stdout must point back at the real stream after a forward, even on raise."""
    real_stdout = sys.stdout
    monitor = make_monitor(model=_OverflowToyModel())
    monitor.set_step(0)
    with pytest.raises(ValueError):
        monitor._hf.model(torch.randn(2, 4))
    assert sys.stdout is real_stdout


def test_monitor__close_is_idempotent(tmp_path: Path, make_monitor) -> None:
    monitor = make_monitor()
    monitor.close()
    monitor.close()  # must not raise


def test_monitor__close_closes_log_file(tmp_path: Path, make_monitor) -> None:
    monitor = make_monitor()
    log_file = monitor._log_file
    monitor.close()
    assert log_file.closed
    with pytest.raises(ValueError):
        log_file.write("after close")


def test_monitor__close_detaches_forward_hooks(tmp_path: Path, make_monitor) -> None:
    """After close(), forward passes on the model must not try to write to the log."""
    model = _FiniteToyModel()
    monitor = make_monitor(model=model)
    assert any(len(m._forward_hooks) > 0 for m in model.modules()), (
        "precondition: hooks must be attached before close()"
    )
    monitor.close()
    assert all(len(m._forward_hooks) == 0 for m in model.modules()), (
        "close() must detach hooks"
    )
    # And a subsequent forward must not raise.
    model(torch.randn(2, 4))


def test_monitor__close_safe_after_forward_raises(tmp_path: Path, make_monitor) -> None:
    """close() must not raise after forward_hook has raised ValueError."""
    monitor = make_monitor(model=_OverflowToyModel())
    monitor.set_step(0)
    with pytest.raises(ValueError):
        monitor._hf.model(torch.randn(2, 4))
    monitor.close()  # must not raise


def test_monitor__context_manager_closes_on_exception(
    tmp_path: Path, make_monitor
) -> None:
    """__exit__ closes the monitor even when the body raises."""
    monitor = make_monitor(model=_OverflowToyModel())
    with monitor as m:
        m.set_step(0)
        with pytest.raises(ValueError):
            m._hf.model(torch.randn(2, 4))
    assert monitor._log_file.closed
    assert all(len(mod._forward_hooks) == 0 for mod in monitor._hf.model.modules())


def test_monitor__init_failure_closes_log_file(tmp_path: Path) -> None:
    """If HF __init__ raises, the log file opened in our __init__ is closed."""

    class _BoomModel(_FiniteToyModel):
        def apply(self, fn, recurse: bool = True):  # type: ignore[override]
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        UnderflowOverflowMonitor(
            model=_BoomModel(),
            debug_args=DebugUnderflowOverflowArgs(enabled=True),
            out_dir=tmp_path,
            global_rank=0,
        )
    # The log file should have been created and closed during exception handling.
    log_path = tmp_path / "debug" / "underflow_overflow_rank0.log"
    assert log_path.exists()


# ---------------------------------------------------------------------------
# Compile-conflict tests
# ---------------------------------------------------------------------------
def test_compile_conflict__disabled_debug_allows_compile() -> None:
    """When debug is disabled, compile settings are untouched."""
    debug = DebugArgs()
    compile_args = TorchCompileArgs(disable=False)
    check_compile_conflict(debug, compile_args)  # no raise


def test_compile_conflict__enabled_debug_with_compile_raises() -> None:
    """When debug is enabled and compile is active, raise."""
    debug = DebugArgs(underflow_overflow=DebugUnderflowOverflowArgs(enabled=True))
    compile_args = TorchCompileArgs(disable=False)
    with pytest.raises(ValueError, match="torch.compile cannot be used"):
        check_compile_conflict(debug, compile_args)


def test_compile_conflict__enabled_debug_with_compile_disabled_ok() -> None:
    """When debug is enabled and compile is disabled, no exception."""
    debug = DebugArgs(underflow_overflow=DebugUnderflowOverflowArgs(enabled=True))
    compile_args = TorchCompileArgs(disable=True)
    check_compile_conflict(debug, compile_args)  # no raise


def test_compile_conflict__enabled_false_with_compile_ok() -> None:
    """An explicitly-disabled debug block is treated like a missing one."""
    debug = DebugArgs(
        underflow_overflow=DebugUnderflowOverflowArgs(
            enabled=False, trace_batch_nums=[1], abort_after_batch_num=5
        )
    )
    compile_args = TorchCompileArgs(disable=False)
    check_compile_conflict(debug, compile_args)  # no raise
