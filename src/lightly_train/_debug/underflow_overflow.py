#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Underflow/overflow debugging for task/fine-tuning training.

Wraps HuggingFace ``transformers.debug_utils.DebugUnderflowOverflow``. The upstream
utility is designed for HuggingFace's own training loop; this subclass adapts it to
LightlyTrain's manual Fabric loop with two changes:

1. Batch numbering is driven explicitly by the LightlyTrain training step. Upstream
   increments the batch counter only when the *root* module's forward hook fires.
   LightlyTrain calls ``TrainModel.training_step(...)`` which invokes submodules
   directly and never calls the root module ``forward``, so upstream counting would
   never advance. :meth:`UnderflowOverflowMonitor.set_step` sets the counter instead.
   Diverges from upstream: a new "Starting batch" header is emitted whenever the
   batch number advances (upstream emits the header exactly once).

2. All output is redirected to a per-rank log file. Upstream uses ``print()`` to
   ``stdout``; redirecting it keeps the per-module min/max dumps out of the main
   ``train.log``.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import torch.nn as nn
from transformers.debug_utils import DebugUnderflowOverflow

from lightly_train._debug.debug_args import (
    DebugArgs,
    DebugUnderflowOverflowArgs,
)
from lightly_train._torch_compile import TorchCompileArgs

logger = logging.getLogger(__name__)


class _LightlyDebugUnderflowOverflow(DebugUnderflowOverflow):
    """DebugUnderflowOverflow with LightlyTrain batch numbering and per-rank logging.

    Inherits upstream's hook-registration, batch-numbering bookkeeping, and
    tensor-analysis machinery; only overrides the hook entry point to (a) drive
    the batch header explicitly and (b) redirect all upstream ``print()`` output
    to a caller-supplied log file.
    """

    def __init__(
        self,
        model: nn.Module,
        log_file: Any,
        max_frames_to_save: int,
        trace_batch_nums: list[int],
        abort_after_batch_num: int | None,
    ) -> None:
        self._log_file = log_file
        # Sentinel used by ``forward_hook`` to detect a new batch; the upstream class
        # only ever sets it via the root-module hook, which never fires here.
        self._prev_batch_number = -1
        super().__init__(
            model=model,
            max_frames_to_save=max_frames_to_save,
            trace_batch_nums=trace_batch_nums,
            abort_after_batch_num=abort_after_batch_num,
        )

    def forward_hook(self, module: Any, input: Any, output: Any) -> None:  # type: ignore[override]
        with contextlib.redirect_stdout(self._log_file):
            try:
                # Emit a batch header whenever the batch number advances. Upstream
                # relies on the root module forward firing to delimit batches; in
                # LightlyTrain's manual loop that never happens, so we drive the
                # batch number explicitly via UnderflowOverflowMonitor.set_step.
                if self.batch_number != self._prev_batch_number:
                    self.batch_start_frame()  # type: ignore[no-untyped-call]
                    self._prev_batch_number = self.batch_number
                self.total_calls += 1

                trace_mode = self.batch_number in self.trace_batch_nums
                if trace_mode:
                    self.reset_saved_frames()  # type: ignore[no-untyped-call]

                self.create_frame(module, input, output)  # type: ignore[no-untyped-call]

                if trace_mode:
                    self.trace_frames()  # type: ignore[no-untyped-call]

                if self.detected_overflow and not trace_mode:
                    self.dump_saved_frames()  # type: ignore[no-untyped-call]
                    raise ValueError(
                        "DebugUnderflowOverflow: inf/nan detected, aborting as "
                        "there is no point running further. Please check the "
                        "debug log file for the activation values prior to this "
                        "event."
                    )

                if (
                    self.abort_after_batch_num is not None
                    and self.batch_number > self.abort_after_batch_num
                ):
                    raise ValueError(
                        f"DebugUnderflowOverflow: aborting after batch "
                        f"{self.batch_number} due to "
                        f"`abort_after_batch_num={self.abort_after_batch_num}`."
                    )
            finally:
                # Always flush so partial dumps are recoverable even if we raise
                # before the next explicit flush downstream.
                self._log_file.flush()


class UnderflowOverflowMonitor:
    """Underflow/overflow monitor wrapping HuggingFace's DebugUnderflowOverflow.

    The monitor registers forward hooks on all submodules of ``model`` and records the
    absolute min/max of every weight, input, and output. When a NaN or inf is detected
    in any of them, the last ``max_frames_to_save`` forward frames are written to the
    debug log file and training is aborted by raising ``ValueError``.

    Use as a context manager to guarantee the log file is closed and the forward
    hooks are detached even when an exception is raised inside the training loop
    (including the ``ValueError`` the monitor itself raises on overflow):

    .. code-block:: python

        with UnderflowOverflowMonitor(model, debug_args, out_dir, global_rank) as m:
            for step in range(steps):
                m.set_step(step)
                train_model.training_step(...)

    Args:
        model:
            The training model to attach forward hooks to.
        debug_args:
            Configuration for the monitor.
        out_dir:
            The training output directory. Debug reports are written to
            ``out_dir/debug/underflow_overflow_rank{global_rank}.log``. ``out_dir``
            is created if it does not exist.
        global_rank:
            The global rank of the process, used to give each process its own log file.
    """

    def __init__(
        self,
        model: nn.Module,
        debug_args: DebugUnderflowOverflowArgs,
        out_dir: Path,
        global_rank: int,
    ) -> None:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        log_path = debug_dir / f"underflow_overflow_rank{global_rank}.log"
        # Append mode so multiple debug runs in the same out dir accumulate.
        log_file = log_path.open("a")

        try:
            self._hf = _LightlyDebugUnderflowOverflow(  # type: ignore[no-untyped-call]
                model=model,
                log_file=log_file,
                max_frames_to_save=debug_args.max_frames_to_save,
                trace_batch_nums=list(debug_args.trace_batch_nums),
                abort_after_batch_num=debug_args.abort_after_batch_num,
            )
        except Exception:
            # Hooks may have been registered before construction failed; clear them
            # so the model is usable again, then close the log file.
            self._detach_hooks(model)
            log_file.close()
            raise

        self._log_file = log_file
        self._log_path = log_path
        self._closed = False

        logger.info(
            f"Underflow/overflow debugging enabled on rank {global_rank}. "
            f"Full reports are written to '{log_path}'."
        )

    def __enter__(self) -> UnderflowOverflowMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def set_step(self, step: int) -> None:
        """Set the current training step used for batch numbering and trace/abort logic.

        Must be called before each forward pass (typically at the top of each
        training step) so that the upstream batch counter is driven by the
        LightlyTrain step instead of the (never-firing) root module forward.
        """
        self._hf.batch_number = step

    def close(self) -> None:
        """Detach forward hooks and close the log file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._detach_hooks(self._hf.model)
        self._log_file.close()

    @staticmethod
    def _detach_hooks(model: nn.Module) -> None:
        for module in model.modules():
            module._forward_hooks.clear()


def check_compile_conflict(
    debug_args: DebugArgs, compile_args: TorchCompileArgs
) -> None:
    """Raise if underflow/overflow debugging and ``torch.compile`` are both active.

    Forward hooks from :class:`DebugUnderflowOverflow` and the graph rewrites done
    by ``torch.compile`` do not interact cleanly. Call this once at the start of
    training (after config validation, before Fabric setup) to fail fast.
    """
    if debug_args.is_underflow_overflow_enabled() and not compile_args.disable:
        raise ValueError(
            "torch.compile cannot be used together with underflow/overflow "
            "debugging. Set torch_compile_args.disable=True or disable "
            "underflow/overflow debugging (debug_args.underflow_overflow"
            ".enabled=False)."
        )
