#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from lightning_fabric import Fabric

from lightly_train._debug.debug_args import NaNCaptureArgs
from lightly_train._debug.nan_capture import (
    NaNCaptureMonitor,
    NaNCaptureState,
    NaNDetectedError,
    NaNReplayResult,
    load_nan_capture,
)

# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------


class _MonitorToyModel(nn.Module):
    """Tiny model used to exercise the monitor's grad scan + buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)


class _ReplayToyModel(nn.Module):
    """Tiny model used to exercise the replay path (load + forward/backward).

    Implements the ``training_step`` method that replay relies on. Kept
    self-contained so the test does not depend on a full task pipeline.
    """

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def training_step(
        self, fabric: object, batch: dict[str, torch.Tensor], step: int
    ) -> _ReplayResult:
        out = self.lin(batch["x"])
        loss = (out * out).sum()
        return _ReplayResult(loss=loss)

    def load_train_state_dict(self, *_args: object, **_kwargs: object) -> Any:
        raise AssertionError("NaNCapture replay must load the raw TrainModel state")


class _ReplayResult:
    """Minimal stand-in for ``TaskStepResult`` carrying only what replay needs."""

    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def _make_monitor(
    model: nn.Module, tmp_path: Path, **overrides: object
) -> NaNCaptureMonitor:
    debug_args = NaNCaptureArgs(enabled=True, **overrides)  # type: ignore[arg-type]
    return NaNCaptureMonitor(
        train_model=model,
        train_model_init_kwargs={},
        debug_args=debug_args,
        out_dir=tmp_path,
        global_rank=0,
        gradient_accumulation_steps=1,
    )


class TestNaNCaptureMonitor:
    def test_disabled_monitor_is_noop(self, tmp_path: Path) -> None:
        model = _MonitorToyModel()
        debug_args = NaNCaptureArgs(enabled=False)
        monitor = NaNCaptureMonitor(
            train_model=model,
            train_model_init_kwargs={},
            debug_args=debug_args,
            out_dir=tmp_path,
            global_rank=0,
            gradient_accumulation_steps=1,
        )
        assert monitor.enabled is False
        monitor.begin_step(0)
        monitor.collect_batch({"x": torch.randn(2, 4)})
        # No NaN set — should be a no-op even if enabled were True; but disabled
        # short-circuits regardless.
        monitor.check_and_capture(model)
        assert not (tmp_path / "debug" / "nan_capture").exists()

    def test_collect_batch__clones_detaches_and_moves_to_cpu(
        self, tmp_path: Path
    ) -> None:
        monitor = _make_monitor(_MonitorToyModel(), tmp_path)
        monitor.begin_step(0)
        live = torch.randn(2, 4, requires_grad=True)
        monitor.collect_batch({"x": live})
        stored = monitor._microbatches[0]["x"]  # type: ignore[index]
        assert stored is not live
        assert stored.grad_fn is None
        # Collected immediately to host RAM so the buffer does not hold
        # accelerator memory until save.
        assert stored.device.type == "cpu"
        # Mutating the stored clone (detached, so in-place is allowed) must
        # not affect the live tensor — proves the buffer holds a true clone.
        stored.fill_(1.0)
        assert not torch.equal(stored, live)
        assert live.requires_grad  # live tensor untouched

    def test_begin_step__resets_buffer(self, tmp_path: Path) -> None:
        model = _MonitorToyModel()
        monitor = _make_monitor(model, tmp_path)
        monitor.begin_step(0)
        monitor.collect_batch({"x": torch.randn(2, 4)})
        monitor.collect_batch({"x": torch.randn(2, 4)})
        # Trigger NaN at step 0; capture must contain step 0 microbatches only.
        model.lin.weight.grad = torch.full_like(model.lin.weight, float("nan"))
        with pytest.raises(NaNDetectedError):
            monitor.check_and_capture(model)
        cap0 = torch.load(
            tmp_path / "debug" / "nan_capture" / "rank0" / "nan_capture.pt",
            weights_only=False,
        )
        assert len(cap0["microbatches"]) == 2

        # Now step 1: buffer must be empty until collect_batch populates it again.
        monitor.begin_step(1)
        assert monitor._microbatches == []
        monitor.collect_batch({"x": torch.randn(2, 4)})
        model.lin.weight.grad = torch.full_like(model.lin.weight, float("nan"))
        with pytest.raises(NaNDetectedError):
            monitor.check_and_capture(model)
        cap1 = torch.load(
            tmp_path / "debug" / "nan_capture" / "rank0" / "nan_capture.pt",
            weights_only=False,
        )
        assert len(cap1["microbatches"]) == 1

    def test_check_and_capture__clean_does_not_raise_or_save(
        self, tmp_path: Path
    ) -> None:
        model = _MonitorToyModel()
        monitor = _make_monitor(model, tmp_path)
        monitor.begin_step(0)
        monitor.collect_batch({"x": torch.randn(2, 4)})
        # No backward called → no .grad set → no NaN.
        monitor.check_and_capture(model)
        assert not (tmp_path / "debug" / "nan_capture" / "rank0").exists()

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_check_and_capture__detects_nonfinite_and_writes_capture(
        self, tmp_path: Path, bad_value: float
    ) -> None:
        model = _MonitorToyModel()
        monitor = _make_monitor(model, tmp_path)
        monitor.begin_step(7)
        monitor.collect_batch({"x": torch.randn(2, 4)})
        nan_param = "lin.weight"
        model.lin.weight.grad = torch.full_like(model.lin.weight, bad_value)

        with pytest.raises(NaNDetectedError) as excinfo:
            monitor.check_and_capture(model)

        capture_path = tmp_path / "debug" / "nan_capture" / "rank0" / "nan_capture.pt"
        assert capture_path.exists()
        assert excinfo.value.capture_path == capture_path
        assert nan_param in excinfo.value.nan_param_names

        payload = torch.load(capture_path, weights_only=False)
        assert set(payload.keys()) == {
            "train_model_state_dict",
            "train_model_class_path",
            "train_model_init_kwargs",
            "microbatches",
            "rng_state",
            "metadata",
        }
        assert payload["metadata"].step == 7
        assert payload["metadata"].rank == 0
        assert payload["metadata"].gradient_accumulation_steps == 1
        assert nan_param in payload["metadata"].nan_param_names
        assert len(payload["microbatches"]) == 1
        # Saved state dict holds clean weights (grad NaN does not touch weights).
        assert torch.equal(
            payload["train_model_state_dict"]["lin.weight"],
            model.lin.weight.detach().cpu(),
        )
        assert "torch" in payload["rng_state"]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _save_capture(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    tmp_path: Path,
    rank: int = 0,
    grad_accum: int = 1,
    step: int = 42,
    *,
    induce_nan_grad: bool = True,
) -> Path:
    """Run the monitor through one step that triggers a capture, return dir."""
    monitor = NaNCaptureMonitor(
        train_model=model,
        train_model_init_kwargs={},
        debug_args=NaNCaptureArgs(enabled=True),
        out_dir=tmp_path,
        global_rank=rank,
        gradient_accumulation_steps=grad_accum,
    )
    monitor.begin_step(step)
    for b in batches:
        monitor.collect_batch(b)
    if induce_nan_grad:
        # Set NaN on the first parameter to trigger capture.
        first_param = next(model.parameters())
        first_param.grad = torch.full_like(first_param, float("nan"))
        with pytest.raises(NaNDetectedError):
            monitor.check_and_capture(model)
    else:
        monitor.check_and_capture(model)
    return tmp_path / "debug" / "nan_capture" / f"rank{rank}"


class TestNaNCaptureReplay:
    def test_load_reconstructs_model_with_saved_weights(self, tmp_path: Path) -> None:
        original = _ReplayToyModel()
        # Touch a parameter so state_dict has well-defined values.
        with torch.no_grad():
            original.lin.weight.fill_(0.5)
            original.lin.bias.fill_(-0.25)
        saved_weight = original.lin.weight.detach().clone()
        saved_bias = original.lin.bias.detach().clone()

        batches = [{"x": torch.randn(2, 4)} for _ in range(2)]
        capture_dir = _save_capture(original, batches, tmp_path, grad_accum=2)

        cap = load_nan_capture(capture_dir, device="cpu")
        assert isinstance(cap, NaNCaptureState)
        assert cap.step == 42
        assert cap.rank == 0
        assert cap.gradient_accumulation_steps == 2
        assert len(cap.batches) == 2
        assert torch.equal(cap.model.lin.weight.detach().cpu(), saved_weight)
        assert torch.equal(cap.model.lin.bias.detach().cpu(), saved_bias)

    def test_replay_runs_forward_backward_without_nan(self, tmp_path: Path) -> None:
        original = _ReplayToyModel()
        batches = [{"x": torch.randn(2, 4)}, {"x": torch.randn(2, 4)}]
        capture_dir = _save_capture(original, batches, tmp_path, grad_accum=2)

        cap = load_nan_capture(capture_dir, device="cpu")
        result = cap.replay()

        assert isinstance(result, NaNReplayResult)
        assert result.step == 42
        assert result.reproduced is False
        assert result.nan_param_names == []
        assert len(result.results) == 2

    def test_replay_reproduces_when_forward_produces_nan(self, tmp_path: Path) -> None:
        original = _ReplayToyModel()
        batches = [{"x": torch.randn(2, 4)}]
        capture_dir = _save_capture(original, batches, tmp_path, grad_accum=1)

        cap = load_nan_capture(capture_dir, device="cpu")
        # Corrupt the loaded model's weight so forward output is non-finite →
        # loss non-finite → backward yields non-finite grads → NaN reproduces.
        with torch.no_grad():
            cap.model.lin.weight.fill_(float("nan"))

        result = cap.replay()

        assert result.reproduced is True
        assert len(result.nan_param_names) >= 1
        assert "lin.weight" in result.nan_param_names
        with pytest.raises(NaNDetectedError):
            result.raise_if_reproduced()


# ---------------------------------------------------------------------------
# Integration: the production path the toy tests missed
# ---------------------------------------------------------------------------
# The unit toys above (a) never go through `fabric.setup` and (b) take **kwargs
# and ignore them, so they cannot catch two production bugs this feature had:
#   1. recording the Fabric wrapper class instead of the real TrainModel class,
#   2. dumping pydantic config objects to dicts (real constructors need objects).
# The integration test below wraps via fabric.setup_module and uses a ctor that
# *requires* a config object (rejects plain dicts), exercising both paths.


@dataclass(frozen=True)
class _ReplayConfig:
    """Config object the integration model ctor requires (not a dict)."""

    scale: float = 1.0


class _ReplayFabricToyModel(nn.Module):
    """Replay model whose ctor *requires* a config object (rejects dicts)."""

    def __init__(self, config: _ReplayConfig, **_kwargs: object) -> None:
        if not isinstance(config, _ReplayConfig):
            raise TypeError(
                f"config must be a _ReplayConfig object, got {type(config)!r}"
            )
        super().__init__()
        self.config = config
        self.lin = nn.Linear(4, 4)

    def training_step(
        self, fabric: object, batch: dict[str, torch.Tensor], step: int
    ) -> _ReplayResult:
        out = self.lin(batch["x"]) * self.config.scale
        loss = (out * out).sum()
        return _ReplayResult(loss=loss)

    def load_train_state_dict(self, *_args: object, **_kwargs: object) -> Any:
        raise AssertionError("NaNCapture replay must load the raw TrainModel state")


class TestNaNCaptureIntegration:
    def test_capture_load_replay_through_fabric_setup(self, tmp_path: Path) -> None:
        # Construct, then wrap exactly like the training loop does
        # (fabric.setup), so the recorded class is _FabricModule — proving the
        # monitor unwraps it to the real model class for capture.
        underlying = _ReplayFabricToyModel(config=_ReplayConfig(scale=2.0))
        with torch.no_grad():
            underlying.lin.weight.fill_(0.5)
            underlying.lin.bias.fill_(-0.25)
        saved_weight = underlying.lin.weight.detach().clone()
        saved_bias = underlying.lin.bias.detach().clone()

        fabric = Fabric(accelerator="cpu", devices=1)
        wrapped = fabric.setup_module(underlying)
        assert wrapped.__class__.__name__ == "_FabricModule"  # sanity: wrapped

        monitor = NaNCaptureMonitor(
            train_model=wrapped,
            # Pass the config OBJECT (real ctors need it, not a dumped dict).
            train_model_init_kwargs={"config": _ReplayConfig(scale=2.0)},
            debug_args=NaNCaptureArgs(enabled=True),
            out_dir=tmp_path,
            global_rank=0,
            gradient_accumulation_steps=1,
        )
        monitor.begin_step(99)
        monitor.collect_batch({"x": torch.randn(2, 4)})
        wrapped.lin.weight.grad = torch.full_like(  # type: ignore[attr-defined]
            wrapped.lin.weight,
            float("nan"),  # type: ignore[attr-defined]
        )
        with pytest.raises(NaNDetectedError):
            monitor.check_and_capture(wrapped)

        capture_dir = tmp_path / "debug" / "nan_capture" / "rank0"
        cap = load_nan_capture(capture_dir, device="cpu")

        # Blocker 1 regression guard: reconstructed model is the real class,
        # not _FabricModule.
        assert isinstance(cap.model, _ReplayFabricToyModel)
        assert cap.model.config.scale == 2.0  # config object survived
        # Blocker 2 regression guard: weights loaded from the (Fabric-wrapped)
        # capture match the underlying model's, via clean state_dict keys.
        assert torch.equal(cap.model.lin.weight.detach().cpu(), saved_weight)
        assert torch.equal(cap.model.lin.bias.detach().cpu(), saved_bias)
        assert cap.step == 99

        # Replay runs forward+backward through the reconstructed (unwrapped)
        # model and reports a clean (non-reproduced) result.
        result = cap.replay()
        assert isinstance(result, NaNReplayResult)
        assert result.step == 99
        assert result.reproduced is False
        assert len(result.results) == 1
