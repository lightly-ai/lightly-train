#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Dict

from pytorch_lightning import Callback, LightningModule, Trainer

from lightly_train._events.event_info import TrainingEventInfo
from lightly_train._events.tracker import track_event


class EventsCallback(Callback):
    """Callback to track training events."""

    def __init__(self, event_info: TrainingEventInfo) -> None:
        super().__init__()
        self.event_info = event_info

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Track training_started"""
        if trainer.global_rank != 0:
            return

        properties: Dict[str, Any] = {
            "method": self.event_info.method,
            "model_name": self.event_info.model,
            "task_type": "ssl_pretraining",
            "epochs": self.event_info.epochs,
            "batch_size": self.event_info.batch_size,
            "devices": self.event_info.devices,
        }
        track_event("training_started", properties)
