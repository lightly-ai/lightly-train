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
from pytorch_lightning.utilities import rank_zero_only
from torch.nn import Module

from lightly_train._commands import common_helpers
from lightly_train._commands.common_helpers import ModelFormat
from lightly_train._configs.config import PydanticConfig


class ModelExportArgs(PydanticConfig):
    every_n_epochs: int = 1


class ModelExport(Callback):
    def __init__(
        self,
        model: Module,
        out_dir: Path,
        every_n_epochs: int = 1,
    ):
        self._model = model
        self._out_dir = out_dir
        self._every_n_epochs = every_n_epochs

    @rank_zero_only  # type: ignore[misc]
    def _safe_export_model(self, export_path: Path) -> None:
        """Export the model to the specified path, deleting any existing file."""
        if export_path.exists():
            export_path.unlink(missing_ok=True)

        common_helpers.export_model(
            model=self._model,
            out=export_path,
            format=ModelFormat.PACKAGE_DEFAULT,
            log_example=False,
        )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.current_epoch % self._every_n_epochs == 0:
            # Delete the previous export if it exists
            export_path = self._out_dir / "exported_last.pt"
            self._safe_export_model(export_path)
