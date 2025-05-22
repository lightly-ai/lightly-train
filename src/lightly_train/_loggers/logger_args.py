#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import Field

from lightly_train._configs.config import PydanticConfig
from lightly_train._loggers.jsonl import JSONLLoggerArgs
from lightly_train._loggers.mlflow import MLFlowLoggerArgs
from lightly_train._loggers.tensorboard import TensorBoardLoggerArgs
from lightly_train._loggers.wandb import WandbLoggerArgs


class LoggerArgs(PydanticConfig):
    jsonl: JSONLLoggerArgs | None = Field(default_factory=JSONLLoggerArgs)
    mlflow: MLFlowLoggerArgs | None = None
    tensorboard: TensorBoardLoggerArgs | None = Field(
        default_factory=TensorBoardLoggerArgs
    )
    wandb: WandbLoggerArgs | None = None
