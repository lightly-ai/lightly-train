#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time
from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar


class DataWaitTQDMProgressBar(TQDMProgressBar):
    """
    Customizes the progress bar to include compute efficiency.

    On CUDA, ``on_train_batch_end`` fires before the GPU finishes the backward
    pass.  Recording ``batch_end_time`` there with ``time.perf_counter()``
    causes the remaining async GPU work to bleed into the next step's
    ``data_time``, inflating ``data_wait`` (e.g. ~0.5 % true → ~60 % reported).

    Fix for CUDA:
        - Bracket each step with CUDA events to get true GPU duration.
        - Measure wall-clock time from step-start to step-start (= GPU + data).
        - Subtract GPU duration to isolate data-loading wait.
        - Events are queried at the next step start, by which point the stream
          has advanced past both events — no CPU stall.

    CPU training falls back to simple wall-clock bookkeeping (synchronous
    compute means ``on_train_batch_end`` timing is accurate).
    """

    def __init__(self) -> None:
        super().__init__(refresh_rate=5)
        self.batch_start_time: float | None = None
        self.batch_end_time: float | None = None  # CPU-only
        self.data_time: float | None = None
        self.batch_time: float | None = None
        # CUDA-only: events bracketing the GPU step.
        self._step_start_event: torch.cuda.Event | None = None
        self._step_end_event: torch.cuda.Event | None = None

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        now = time.perf_counter()

        if trainer.strategy.root_device.type == "cuda":
            # 1. Resolve previous step's GPU duration from events
            if self._step_start_event is not None and self._step_end_event is not None:
                self.batch_time = (
                    self._step_start_event.elapsed_time(self._step_end_event) / 1e3  # type: ignore[no-untyped-call]
                )

            # 2. data_time = wall gap (start -> start) minus GPU duration
            if self.batch_start_time is not None and self.batch_time is not None:
                wall_gap = now - self.batch_start_time
                self.data_time = max(0.0, wall_gap - self.batch_time)

            # 3. Begin new step: record start event
            self._step_start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._step_start_event.record()  # type: ignore[no-untyped-call]
        else:
            # CPU: data_time is simply the gap between previous step end and now.
            if self.batch_end_time is not None:
                self.data_time = now - self.batch_end_time

        self.batch_start_time = now
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.strategy.root_device.type == "cuda":
            # Record end event — queried at next step start
            self._step_end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            self._step_end_event.record()  # type: ignore[no-untyped-call]
        elif self.batch_start_time is not None:
            # CPU: batch_time is just wall-clock step duration
            now = time.perf_counter()
            if self.batch_start_time is not None:
                self.batch_time = now - self.batch_start_time
            self.batch_end_time = now
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def get_metrics(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        if self.batch_time is not None and self.data_time is not None:
            if self.batch_time + self.data_time > 0:
                data_wait = self.data_time / (self.batch_time + self.data_time)
                metrics["data_wait"] = f"{data_wait:.1%}"
        return metrics
