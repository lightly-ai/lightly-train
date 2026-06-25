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

2. All output is redirected to a per-rank log file. Upstream uses ``print()`` to
   ``stdout``; redirecting it keeps the per-module min/max dumps out of the main
   ``train.log``.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from lightly_train._debug.debug_args import DebugUnderflowOverflowArgs

logger = logging.getLogger(__name__)


class UnderflowOverflowMonitor:
    """Underflow/overflow monitor wrapping HuggingFace's DebugUnderflowOverflow.

    The monitor registers forward hooks on all submodules of ``model`` and records the
    absolute min/max of every weight, input, and output. When a NaN or inf is detected
    in any of them, the last ``max_frames_to_save`` forward frames are written to the
    debug log file and training is aborted by raising ``ValueError``.

    Args:
        model:
            The training model to attach forward hooks to.
        debug_args:
            Configuration for the monitor.
        out_dir:
            The training output directory. Debug reports are written to
            ``out_dir/debug/underflow_overflow_rank{global_rank}.log``.
        global_rank:
            The global rank of the process, used to give each process its own log file.
    """

    def __init__(
        self,
        model: Any,
        debug_args: DebugUnderflowOverflowArgs,
        out_dir: Path,
        global_rank: int,
    ) -> None:
        # Lazy import because transformers is an optional dependency. Importing here
        # (instead of at module level) keeps training without debugging independent of
        # transformers being importable.
        from transformers.debug_utils import DebugUnderflowOverflow

        self._global_rank = global_rank
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = debug_dir / f"underflow_overflow_rank{global_rank}.log"
        # Open in append mode so multiple debug runs in the same out dir accumulate.
        self._log_file = self._log_path.open("a")

        monitor = self

        class _LightlyDebugUnderflowOverflow(DebugUnderflowOverflow):
            """DebugUnderflowOverflow with LightlyTrain batch numbering and logging."""

            # Track the batch number whose header has already been emitted so that a
            # new "Starting batch" header is written whenever set_step advances it.
            _prev_batch_number: int = -1

            def forward_hook(self, module: Any, input: Any, output: Any) -> None:
                # Redirect all upstream print() output to the per-rank debug log file.
                with contextlib.redirect_stdout(monitor._log_file):
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
                        monitor._log_file.flush()

                    if self.detected_overflow and not trace_mode:
                        self.dump_saved_frames()  # type: ignore[no-untyped-call]
                        # Flush before raising so the report is on disk.
                        monitor._log_file.flush()
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

        # super().__init__() registers forward hooks pointing at the overridden
        # forward_hook above, so hooks call our LightlyTrain version from the start.
        self._hf = _LightlyDebugUnderflowOverflow(  # type: ignore[no-untyped-call]
            model=model,
            max_frames_to_save=debug_args.max_frames_to_save,
            trace_batch_nums=list(debug_args.trace_batch_nums),
            abort_after_batch_num=debug_args.abort_after_batch_num,
        )

        logger.info(
            f"Underflow/overflow debugging enabled on rank {global_rank}. "
            f"Full reports are written to '{self._log_path}'."
        )

    def set_step(self, step: int) -> None:
        """Set the current training step used for batch numbering and trace/abort logic."""
        self._hf.batch_number = step

    def close(self) -> None:
        self._log_file.close()
