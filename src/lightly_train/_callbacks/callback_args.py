#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import Field

from lightly_train._callbacks.checkpoint import ModelCheckpointArgs
from lightly_train._callbacks.export import ModelExportArgs
from lightly_train._configs.config import PydanticConfig


class LearningRateMonitorArgs(PydanticConfig):
    pass


class DeviceStatsMonitorArgs(PydanticConfig):
    pass


class EarlyStoppingArgs(PydanticConfig):
    monitor: str = "train_loss"
    patience: int = int(1e12)
    check_finite: bool = True


class CallbackArgs(PydanticConfig):
    learning_rate_monitor: LearningRateMonitorArgs | None = Field(
        default_factory=LearningRateMonitorArgs
    )
    device_stats_monitor: DeviceStatsMonitorArgs | None = None
    early_stopping: EarlyStoppingArgs | None = Field(default_factory=EarlyStoppingArgs)
    model_export: ModelExportArgs | None = Field(default_factory=ModelExportArgs)
    model_checkpoint: ModelCheckpointArgs | None = Field(
        default_factory=ModelCheckpointArgs
    )
