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

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar


class DataWaitTQDMProgressBar(TQDMProgressBar):
    """
    Customizes the progress bar to include compute efficiency.
    """

    def __init__(self) -> None:
        super().__init__(refresh_rate=5)
        self.batch_start_time: float | None = None
        self.batch_end_time: float | None = None
        self.data_time: float | None = None
        self.batch_time: float | None = None

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.batch_start_time = time.perf_counter()
        if self.batch_end_time is not None:
            self.data_time = self.batch_start_time - self.batch_end_time
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.batch_end_time = time.perf_counter()
        if self.batch_start_time is not None:
            self.batch_time = self.batch_end_time - self.batch_start_time
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
