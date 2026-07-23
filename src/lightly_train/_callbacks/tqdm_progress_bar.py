#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar


class DataWaitTQDMProgressBar(TQDMProgressBar):
    """Customizes the progress bar to include compute efficiency."""

    def __init__(self) -> None:
        super().__init__(refresh_rate=5)

    def get_metrics(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        metrics = super().get_metrics(trainer, pl_module)
        batch_time = trainer.callback_metrics.get("profiling/batch_time")
        data_time = trainer.callback_metrics.get("profiling/data_time")
        if batch_time is not None and data_time is not None:
            if batch_time + data_time > 0:
                data_wait = data_time / (batch_time + data_time)
                metrics["data_wait"] = f"{data_wait:.1%}"
        return metrics
