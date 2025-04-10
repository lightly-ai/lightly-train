#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.nn import Module

from lightly_train._callbacks.callback_args import (
    CallbackArgs,
)
from lightly_train._callbacks.checkpoint import ModelCheckpoint
from lightly_train._callbacks.tqdm_progress_bar import DataWaitTQDMProgressBar
from lightly_train._checkpoint import CheckpointLightlyTrainModels
from lightly_train._configs import validate
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.feature_extractor import ModelGetter
from lightly_train._transforms.transform import NormalizeArgs


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
    model: Module,
    embedding_model: EmbeddingModel,
) -> list[Callback]:
    callbacks: list[Callback] = []
    callbacks.append(DataWaitTQDMProgressBar())
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
    if callback_args.model_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                models=CheckpointLightlyTrainModels(
                    model=model, embedding_model=embedding_model
                ),
                dirpath=out / "checkpoints",
                normalize_args=normalize_args,
                **callback_args.model_checkpoint.model_dump(),
            )
        )
    return callbacks


def get_checkpoint_model(model: Module) -> Module:
    # If feature extractor provides .model() getter, store only the model.
    if isinstance(model, ModelGetter):
        model_: Module = model.get_model()
        return model_
    return model
