#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import Logger

from lightly_train._callbacks.callback_args import (
    CallbackArgs,
)
from lightly_train._callbacks.checkpoint import ModelCheckpoint
from lightly_train._callbacks.export import ModelExport
from lightly_train._callbacks.mlflow_logging import MLFlowLogging
from lightly_train._callbacks.tqdm_progress_bar import DataWaitTQDMProgressBar
from lightly_train._checkpoint import CheckpointLightlyTrainModels
from lightly_train._configs import validate
from lightly_train._loggers.jsonl import JSONLLogger
from lightly_train._loggers.mlflow import MLFlowLogger
from lightly_train._loggers.tensorboard import TensorBoardLogger
from lightly_train._loggers.wandb import WandbLogger
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._transforms.transform import NormalizeArgs

AnyLoggerType = TypeVar(
    "AnyLoggerType",
    Logger,
    JSONLLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
    None,
)


def get_callback_args(
    callback_args: dict[str, Any] | CallbackArgs | None,
) -> CallbackArgs:
    if isinstance(callback_args, CallbackArgs):
        return callback_args
    callback_args = {} if callback_args is None else callback_args
    return validate.pydantic_model_validate(CallbackArgs, callback_args)


def get_callbacks(
    callback_args: CallbackArgs,
    normalize_args: NormalizeArgs,
    out: Path,
    wrapped_model: ModelWrapper,
    embedding_model: EmbeddingModel,
    loggers: list[AnyLoggerType],
) -> list[Callback]:
    callbacks: list[Callback] = []
    callbacks.append(DataWaitTQDMProgressBar())
    if any(isinstance(c, MLFlowLogger) for c in loggers):
        callbacks.append(MLFlowLogging())
    if callback_args.learning_rate_monitor is not None:
        callbacks.append(
            LearningRateMonitor(**callback_args.learning_rate_monitor.model_dump())
        )
    if callback_args.device_stats_monitor is not None:
        callbacks.append(
            DeviceStatsMonitor(**callback_args.device_stats_monitor.model_dump())
        )
    if callback_args.early_stopping is not None:
        callbacks.append(EarlyStopping(**callback_args.early_stopping.model_dump()))
    if callback_args.model_export is not None:
        callbacks.append(
            ModelExport(
                wrapped_model=wrapped_model,
                out_dir=out / "exported_models",
                **callback_args.model_export.model_dump(),
            )
        )
    if callback_args.model_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                models=CheckpointLightlyTrainModels(
                    model=wrapped_model.get_model(),
                    wrapped_model=wrapped_model,
                    embedding_model=embedding_model,
                ),
                dirpath=out / "checkpoints",
                normalize_args=normalize_args,
                **callback_args.model_checkpoint.model_dump(),
            )
        )
    return callbacks
