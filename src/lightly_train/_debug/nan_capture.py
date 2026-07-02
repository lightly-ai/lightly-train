#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""NaN/Inf capture for reproducing bad batches in fine-tuning training.

Self-contained capture + replay tool for the manual Fabric training loop
(``lightly_train._commands.train_task``).

**Scheme.** When enabled via ``DebugArgs.nancapture.enabled=True``:

- On each training step, :class:`NaNCaptureMonitor` clones+detaches each
  microbatch into a per-step buffer and snapshots the torch (and CUDA, if
  available) RNG state.
- After the gradient-accumulation loop completes and *before*
  ``clip_gradients``/``optimizer.step()`` (the point at which a NaN gradient
  would corrupt the model via the optimizer step), the monitor scans all
  parameter ``.grad`` tensors for NaN/Inf.
- On detection, the monitor writes a single self-contained
  ``out_dir/debug/nan_capture/rank{R}/nan_capture.pt`` holding the model
  state dict, the TrainModel class path + init kwargs for reconstruction,
  the step's microbatches, RNG state, and metadata — then raises
  :class:`NaNDetectedError` to halt training.

The standard ``checkpoints/last.ckpt`` is *not* touched; ``resume_interrupted``
from ``out_dir`` is unaffected.

**Replay.** Capture files are reproducibly loadable via
:func:`load_nan_capture` + :meth:`NaNCaptureState.replay`. The replay
reconstructs the TrainModel, restores RNG, and re-runs the triggering
forward+backward sequence — zero-setup (auto-creates a single-device Fabric
if none is passed). Stops before ``clip_gradients``/``optimizer.step`` since
the NaN lives in gradients/activations, and the corruption path is the
optimizer step which the training loop never reached.

Caveat: replay uses default Fabric precision (``"32-true"``). To reproduce a
mixed-precision failure (e.g. bfloat16 overflow), construct your own Fabric
matching the captured run's precision and pass it: ``cap.replay(fabric=f)``.

Known limitation: the capture holds the model state **after** the triggering
step's forward/backward. Model **parameters** are unaffected by forward/backward
(they are only changed by the optimizer step, which never ran), so weight-based
reproduction is faithful. But training-mode **buffers** mutated during the
forward pass (e.g. BatchNorm running stats) are captured at their post-batch
values, not pre-batch — so replay may start from slightly stale buffers. This is
accepted for v1; it does not affect dropout/data-driven NaN debugging. If the
suspect path runs through BatchNorm-style buffers, treat replay results with
that caveat.

**References.** The capture-and-replay scheme (clone microbatches, snapshot RNG,
write the capture before the optimizer step, halt on NaN/Inf) follows Chaim Rand,
"Debugging the Dreaded NaN — Capturing and Reproducing Failures in PyTorch
Training with Lightning", Feb 2025.
https://chaimrand.medium.com/debugging-the-dreaded-nan-ac3f9feac5b2
"""

from __future__ import annotations

import datetime
import importlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn import Module

from lightly_train._debug.debug_args import NaNCaptureArgs

logger = logging.getLogger(__name__)

_CAPTURE_FILENAME = "nan_capture.pt"


class NaNDetectedError(RuntimeError):
    """Raised when a NaN/Inf is detected in parameter gradients.

    The capture (model state, microbatches, RNG, metadata) has already been
    written to ``capture_path`` before this exception is raised. To reproduce
    the failure::

        from lightly_train._debug.nan_capture import load_nan_capture
        cap = load_nan_capture("<capture_path>")
        cap.replay()
    """

    def __init__(self, nan_param_names: list[str], capture_path: Path) -> None:
        self.nan_param_names = list(nan_param_names)
        self.capture_path = Path(capture_path)
        first = self.nan_param_names[0] if self.nan_param_names else "<unknown>"
        super().__init__(
            f"NaN/Inf detected in gradients at {len(self.nan_param_names)} "
            f"parameter(s). First failing parameter: '{first}'. Capture saved "
            f"to '{self.capture_path}'. Reproduce with: "
            f"lightly_train._debug.nan_capture.load_nan_capture("
            f"'{self.capture_path}').replay()."
        )


@dataclass
class NaNCaptureMetadata:
    """Metadata for a captured NaN/Inf occurrence."""

    step: int
    rank: int
    timestamp: str
    nan_param_names: list[str]
    gradient_accumulation_steps: int
    train_model_class_path: str
    train_model_device: str


@dataclass
class NaNReplayResult:
    """Result of replaying a captured NaN/Inf.

    Attributes:
        step: Training step that was replayed.
        results: Per-microbatch ``TaskStepResult`` outputs (in order).
        nan_param_names: Parameter names whose ``.grad`` is NaN/Inf after the
            replayed accumulation; empty if the NaN did not reproduce.
        reproduced: ``True`` iff ``nan_param_names`` is non-empty.
    """

    step: int
    results: list[Any]
    nan_param_names: list[str]
    reproduced: bool

    def raise_if_reproduced(self) -> None:
        """Re-raise :class:`NaNDetectedError` if the NaN reproduced."""
        if self.reproduced:
            raise NaNDetectedError(
                nan_param_names=self.nan_param_names,
                capture_path=Path("(replayed capture)"),
            )


@dataclass
class NaNCaptureState:
    """Loaded NaN capture, ready for replay.

    Construct via :func:`load_nan_capture`. Call :meth:`replay` to reproduce
    the captured failure with zero setup.
    """

    step: int
    rank: int
    gradient_accumulation_steps: int
    model: Any  # TrainModel-like; replay requires ``training_step``.
    batches: list[Any]
    rng_state: dict[str, Any]
    metadata: NaNCaptureMetadata

    def replay(self, fabric: Fabric | None = None) -> NaNReplayResult:
        """Re-run the triggering step with saved microbatches and RNG.

        Zero-setup: if ``fabric`` is None, creates a single-device Fabric
        (``accelerator="auto"``, ``devices=1``) — no ``launch()`` since replay
        runs in the main process. Mirrors the training loop exactly: restores
        RNG, runs ``training_step`` + ``fabric.backward(loss / grad_accum)``
        over saved microbatches (with ``fabric.no_backward_sync`` on all but
        the last microbatch), then scans grads. Stops before
        ``clip_gradients``/``optimizer.step`` (the corruption path that the
        training loop never reached).

        Caveat: auto-Fabric uses default precision (``"32-true"``). To
        reproduce a mixed-precision failure, pass a Fabric matching the
        captured run's precision.
        """
        if fabric is None:
            fabric = Fabric(accelerator="auto", devices=1)

        _restore_rng(self.rng_state)

        self.model.train()
        model = fabric.setup_module(self.model)

        device = next(model.parameters()).device
        grad_accum = self.gradient_accumulation_steps

        results: list[Any] = []
        for acc_step in range(grad_accum):
            is_accumulating = acc_step < grad_accum - 1
            batch = _batch_to_device(self.batches[acc_step], device)
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                result = model.training_step(fabric=fabric, batch=batch, step=self.step)
                fabric.backward(result.loss / grad_accum)
            results.append(result)

        nan_names = _scan_grads_nan(model)
        return NaNReplayResult(
            step=self.step,
            results=results,
            nan_param_names=nan_names,
            reproduced=bool(nan_names),
        )


class NaNCaptureMonitor:
    """In-training monitor that captures training state on NaN/Inf detection.

    Wired into the fine-tuning training loop after gradient accumulation,
    before ``clip_gradients``/``optimizer.step()``. All hooks are no-ops when
    :attr:`enabled` is False.

    Lifecycle::

        with NaNCaptureMonitor(...) as monitor:
            for step in range(steps):
                monitor.begin_step(step)
                for acc_step in range(grad_accum):
                    batch = next(dataloader)
                    monitor.collect_batch(batch)
                    # ... training_step + backward ...
                monitor.check_and_capture(train_model)  # raises on NaN
    """

    def __init__(
        self,
        train_model: Module,
        train_model_init_kwargs: Mapping[str, object],
        debug_args: NaNCaptureArgs,
        out_dir: Path,
        global_rank: int,
        gradient_accumulation_steps: int,
    ) -> None:
        self._train_model = train_model
        self._train_model_class_path = _get_model_class_path(train_model)
        self._train_model_init_kwargs = dict(train_model_init_kwargs)
        self._out_dir = Path(out_dir)
        self._rank = global_rank
        self._grad_accum = gradient_accumulation_steps
        self._enabled = bool(debug_args.enabled)
        self._microbatches: list[Any] = []
        self._step: int | None = None
        self._rng_state: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __enter__(self) -> NaNCaptureMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release per-step buffered state. Idempotent."""
        self._microbatches = []
        self._rng_state = {}
        self._step = None

    def begin_step(self, step: int) -> None:
        if not self._enabled:
            return
        self._step = step
        self._microbatches = []
        self._rng_state = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            device = next(self._train_model.parameters()).device
            if device.type == "cuda":
                self._rng_state["cuda"] = torch.cuda.get_rng_state(device)

    def collect_batch(self, batch: Any) -> None:
        if not self._enabled:
            return
        # Clone + detach + move to CPU immediately so the per-step buffer lives
        # on host RAM (not accelerator) and is decoupled from the live autograd
        # graph. Keeps debug-time accelerator memory bounded to the forward
        # pass instead of holding the whole microbatch set until save.
        self._microbatches.append(_clone_batch_to_cpu(batch))

    def check_and_capture(self, train_model: Module) -> None:
        if not self._enabled:
            return
        nan_names = _scan_grads_nan(train_model)
        if not nan_names:
            return
        capture_dir = self._out_dir / "debug" / "nan_capture" / f"rank{self._rank}"
        capture_dir.mkdir(parents=True, exist_ok=True)
        capture_path = capture_dir / _CAPTURE_FILENAME

        state_dict_cpu = _state_dict_to_cpu(train_model.state_dict())
        microbatches_cpu = list(self._microbatches)
        device_str = str(next(train_model.parameters()).device)
        metadata = NaNCaptureMetadata(
            step=self._step if self._step is not None else -1,
            rank=self._rank,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            nan_param_names=nan_names,
            gradient_accumulation_steps=self._grad_accum,
            train_model_class_path=self._train_model_class_path,
            train_model_device=device_str,
        )

        payload = {
            "train_model_state_dict": state_dict_cpu,
            "train_model_class_path": self._train_model_class_path,
            "train_model_init_kwargs": self._train_model_init_kwargs,
            "microbatches": microbatches_cpu,
            "rng_state": self._rng_state,
            "metadata": metadata,
        }
        # Atomic write: save to a temp file then os.replace onto the final
        # path so a crash/disk-full/interruption can never leave a partially
        # written nan_capture.pt (which would be worse than no capture).
        tmp_path = capture_path.with_suffix(".pt.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, capture_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        logger.error(
            f"NaN/Inf captured at step {self._step} on rank {self._rank}: "
            f"{len(nan_names)} NaN/Inf parameter(s). Capture saved to "
            f"'{capture_path}'."
        )
        raise NaNDetectedError(nan_param_names=nan_names, capture_path=capture_path)


def load_nan_capture(capture_dir: Path, device: str = "cpu") -> NaNCaptureState:
    """Load a NaN capture from ``capture_dir`` for replay.

    ``capture_dir`` is the rank directory (e.g.
    ``out_dir/debug/nan_capture/rank0``). Reconstructs the TrainModel from
    the captured class path + init kwargs (with ``load_weights=False`` since
    the captured state dict supplies the weights), loads the saved raw TrainModel
    state dict strictly, restores the saved microbatches and RNG state. Does
    not run anything — call :meth:`NaNCaptureState.replay` to reproduce the
    failure.
    """
    capture_dir = Path(capture_dir)
    capture_path = capture_dir / _CAPTURE_FILENAME
    if not capture_path.is_file():
        raise FileNotFoundError(f"NaN capture file not found at '{capture_path}'.")

    payload = torch.load(capture_path, map_location=device, weights_only=False)
    state_dict = payload["train_model_state_dict"]
    class_path = payload["train_model_class_path"]
    init_kwargs = dict(payload["train_model_init_kwargs"])
    microbatches = payload["microbatches"]
    rng_state = payload["rng_state"]
    metadata = payload["metadata"]

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    train_model_cls = getattr(module, class_name)
    # The captured state dict supplies weights; don't re-download the backbone.
    init_kwargs["load_weights"] = False
    model = train_model_cls(**init_kwargs)
    # A capture is meant to be a self-contained, faithful TrainModel snapshot:
    # replay must restore exactly the raw state that produced the bad gradients.
    # Do not use task-specific checkpoint/export loaders here; some prefer EMA
    # weights when present, while the live model weights produced the NaN.
    model.load_state_dict(state_dict, strict=True)
    logger.info(
        "Loaded NaN capture from '%s' (step %s, rank %s).",
        capture_path,
        metadata.step,
        metadata.rank,
    )
    model.to(device)

    return NaNCaptureState(
        step=metadata.step,
        rank=metadata.rank,
        gradient_accumulation_steps=metadata.gradient_accumulation_steps,
        model=model,
        batches=microbatches,
        rng_state=rng_state,
        metadata=metadata,
    )


# ---- helpers ----------------------------------------------------------------


def _get_model_class_path(model: Module) -> str:
    """Return the import path for the real TrainModel behind Fabric wrappers."""
    unwrapped = _unwrap_train_model(model)
    return f"{unwrapped.__class__.__module__}.{unwrapped.__class__.__qualname__}"


def _unwrap_train_model(model: Module) -> Module:
    """Return the original TrainModel for Lightning Fabric-wrapped modules."""
    if model.__class__.__name__ == "_FabricModule":
        unwrapped = getattr(model, "module", model)
        if isinstance(unwrapped, Module):
            return unwrapped
    return model


def _clone_batch_to_cpu(batch: Any) -> Any:
    """Clone + detach + move to CPU all tensors in ``batch``.

    Recurses dicts/lists/tuples. Cloning+detaching breaks autograd links with
    the live forward pass; moving to CPU keeps the per-step microbatch buffer
    in host RAM rather than holding accelerator memory until the capture save.
    """
    if isinstance(batch, torch.Tensor):
        return batch.detach().clone().cpu()
    if isinstance(batch, dict):
        return {k: _clone_batch_to_cpu(v) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_clone_batch_to_cpu(v) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_clone_batch_to_cpu(v) for v in batch)
    return batch


def _batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_batch_to_device(v, device) for v in batch)
    return batch


def _state_dict_to_cpu(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    return {k: (v.cpu() if isinstance(v, Tensor) else v) for k, v in state_dict.items()}


def _scan_grads_nan(model: Module) -> list[str]:
    """Return names of parameters whose ``.grad`` contains NaN or Inf.

    Strips the ``_forward_module.`` prefix that Lightning Fabric's
    ``_FabricModule`` adds to ``named_parameters()``, so reported names
    match the user's model layout (e.g. ``lin.weight`` rather than
    ``_forward_module.lin.weight``).
    """
    nan_names: list[str] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            clean = name
            if clean.startswith("_forward_module."):
                clean = clean[len("_forward_module.") :]
            nan_names.append(clean)
    return nan_names


def _restore_rng(rng_state: dict[str, Any]) -> None:
    torch.set_rng_state(rng_state["torch"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state["cuda"])
