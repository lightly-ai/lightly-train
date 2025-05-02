#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.nn import Module

from lightly_train._configs.config import PydanticConfig
from lightly_train._commands import common_helpers
from lightly_train._commands.common_helpers import ModelFormat


class ModelExportArgs(PydanticConfig):
    every_n_epochs: int | None = None


class ModelExport(Callback):
    def __init__(
        self,
        model: Module,
        out_dir: Path,
        every_n_epochs: int | None,
    ):
        self._model = model
        self._out_dir = out_dir
        if every_n_epochs is None:
            every_n_epochs = 1
        self._every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.current_epoch % self._every_n_epochs == 0:
            # Delete the previous export if it exists
            export_path = self._out_dir / "exported_last.pt"
            if export_path.exists():
                export_path.unlink()
            
            common_helpers.export_model(
                model=self._model,
                out=export_path,
                format=ModelFormat.PACKAGE_DEFAULT,
                log_example=False,
            )
