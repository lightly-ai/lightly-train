#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

from pydantic import Field

from lightly_train._configs.config import PydanticConfig
from lightly_train._loggers.mlflow import MLFlowLoggerArgs
from lightly_train._loggers.tensorboard import TensorBoardLoggerArgs


class TaskLoggerArgs(PydanticConfig):
    log_every_num_steps: int | Literal["auto"] = "auto"
    val_every_num_steps: int | Literal["auto"] = "auto"
    val_log_every_num_steps: int | Literal["auto"] = "auto"

    mlflow: MLFlowLoggerArgs | None = None
    tensorboard: TensorBoardLoggerArgs | None = Field(
        default_factory=TensorBoardLoggerArgs
    )

    def resolve_auto(self, steps: int, val_steps: int) -> None:
        if self.log_every_num_steps == "auto":
            self.log_every_num_steps = min(100, max(1, steps // 10))
        if self.val_every_num_steps == "auto":
            self.val_every_num_steps = min(1000, max(1, steps))
        if self.val_log_every_num_steps == "auto":
            self.val_log_every_num_steps = min(20, max(1, val_steps))
