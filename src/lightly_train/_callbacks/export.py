#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations
from pathlib import Path

from pytorch_lightning.callbacks import Callback
from src.lightly_train._commands.common_helpers import export_model, ModelFormat
from torch.nn import Module


class ModelExport(Callback):
    def __init__(
        self,
        model: Module,
        out_dir: Path,
    ):
        self._model = model
        self._out_dir = out_dir
    
    def on_train_epoch_end(self, trainer, pl_module):
        export_model(
            model=self._model,
            out=self._out_dir / "exported_models" / "exported_last.pt",
            format=ModelFormat.PACKAGE_DEFAULT,
        )