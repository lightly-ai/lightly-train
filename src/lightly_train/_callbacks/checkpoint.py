#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint

from lightly_train._checkpoint import (
    CHECKPOINT_LIGHTLY_TRAIN_KEY,
    CheckpointLightlyTrain,
    CheckpointLightlyTrainModels,
)
from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class ModelCheckpointArgs(PydanticConfig):
    auto_insert_metric_name: bool = True
    enable_version_counter: bool = False
    every_n_epochs: int | None = None
    every_n_train_steps: int | None = None
    filename: str | None = None
    mode: str = "min"
    monitor: str | None = None
    save_last: bool = True
    save_on_train_epoch_end: bool | None = None
    save_top_k: int = 1
    save_weights_only: bool = False


class ModelCheckpoint(_ModelCheckpoint):
    def __init__(
        self,
        models: CheckpointLightlyTrainModels,
        normalize_args: NormalizeArgs,
        dirpath: None | PathLike = None,
        filename: None | str = None,
        monitor: None | str = None,
        verbose: bool = False,
        # Note: the type of save_last depends on the version of pytorch_lightning.
        # Only later versions also allow Literal["link"] as type.
        save_last: None | bool = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: None | int = None,
        train_time_interval: None | timedelta = None,
        every_n_epochs: None | int = None,
        save_on_train_epoch_end: None | bool = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )
        self._models = models
        self._normalize_args = normalize_args

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        checkpoint[CHECKPOINT_LIGHTLY_TRAIN_KEY] = CheckpointLightlyTrain.from_now(
            models=self._models,
            normalize_args=self._normalize_args,
        ).to_dict()

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        super().on_load_checkpoint(trainer, pl_module, checkpoint)
        try:
            _checkpoint = CheckpointLightlyTrain.from_checkpoint_dict(checkpoint)
        except KeyError as ex:
            logger.warning(
                f"Could not restore lightly_train models from checkpoint: {ex}"
            )
        else:
            # Load the state dicts from the checkpoint in-place.
            self._models.model.load_state_dict(_checkpoint.models.model.state_dict())
            self._models.wrapped_model.load_state_dict(
                _checkpoint.models.wrapped_model.state_dict()
            )
            self._models.embedding_model.load_state_dict(
                _checkpoint.models.embedding_model.state_dict()
            )

            # Raise a warning if the normalize_args do not match.
            if self._normalize_args != _checkpoint.normalize_args:
                logger.warning(
                    f"`normalize_args` mismatch:\n"
                    f"  Current:    {self._normalize_args}\n"
                    f"  Checkpoint: {_checkpoint.normalize_args}"
                )
