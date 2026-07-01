#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Underflow/overflow debugging for task/fine-tuning training.

Thin wrapper around the vendored :class:`DebugUnderflowOverflow` (see
:mod:`lightly_train._debug.huggingface_debug_utils`). The vendored class is
configured in ``batch_number_mode="manual"`` so the LightlyTrain step drives
``batch_number`` explicitly (LightlyTrain's manual Fabric loop never calls
the root module's ``forward`` so the upstream auto-increment would never
fire). Output is redirected to a per-rank log file via the ``log_file`` arg
so per-module min/max dumps do not pollute the main ``train.log``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.nn import Module

from lightly_train._debug.debug_args import (
    DebugArgs,
    DebugUnderflowOverflowArgs,
)
from lightly_train._debug.huggingface_debug_utils import DebugUnderflowOverflow
from lightly_train._torch_compile import TorchCompileArgs

logger = logging.getLogger(__name__)


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
        model: Module,
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
            self._hf = DebugUnderflowOverflow(
                model=model,
                max_frames_to_save=debug_args.max_frames_to_save,
                trace_batch_nums=list(debug_args.trace_batch_nums),
                abort_after_batch_num=debug_args.abort_after_batch_num,
                log_file=log_file,
                batch_number_mode="manual",
            )
        except Exception:
            # ``DebugUnderflowOverflow.__init__`` self-cleans via
            # ``detach_hooks`` on partial-failure (e.g. ``model.apply`` raising
            # mid-registration); here we only need to close the log file we
            # opened.
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
        training step) so the vendored monitor's batch counter is driven by the
        LightlyTrain step instead of the (never-firing) root module forward.
        """
        self._hf.set_batch_number(step)

    def close(self) -> None:
        """Detach forward hooks and close the log file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._hf.detach_hooks()
        self._log_file.close()


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
