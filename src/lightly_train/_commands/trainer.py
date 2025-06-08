#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from lightning_fabric import Fabric
from lightning_fabric.wrappers import (
    _FabricDataLoader,
    _FabricModule,
    _FabricOptimizer,
)
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from pytorch_lightning.core.hooks import DataHooks
from pytorch_lightning.callbacks import ProgressBar


class LightlyTrainTrainer:
    def __init__(
        self,
        *,
        fabric: Fabric,
        max_epochs: int,
        log_every_n_steps: int,
        default_root_dir: Path,
    ) -> None:
        self._fabric = fabric
        self._max_epochs = max_epochs
        self._log_every_n_steps = log_every_n_steps
        self._default_root_dir = default_root_dir

    @property
    def accelerator(self):
        return self._fabric.accelerator
    
    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def strategy(self):
        return self._fabric.strategy

    @property
    def world_size(self):
        return self._fabric.world_size

    @property
    def num_nodes(self):
        return (
            self._fabric.strategy.num_nodes
            if hasattr(self._fabric.strategy, "num_nodes")
            else 1
        )

    @property
    def num_devices(self):
        return self.world_size // self.num_nodes
    
    @property
    def is_global_zero(self) -> bool:
        return self._fabric.is_global_zero
    
    @property
    def loggers(self):
        return self._fabric.loggers
    
    @property
    def val_check_interval(self) -> int:
        return 1
    
    @property
    def check_val_every_n_epoch(self) -> int:
        return 1
    
    @property
    def lr_scheduler_configs(self):
        return self._lr_scheduler_configs

    def fit(
        self,
        model: LightningModule,
        dataloader: DataLoader,
        ckpt_path: str | None = None,
    ) -> None:
        self._len_dataloader = len(dataloader)
        model._trainer = self
        optimizers, schedulers = model.configure_optimizers()
        self._lr_scheduler_configs = schedulers
        model, optimizers = self._fabric.setup(model, *optimizers)
        dataloader = self._fabric.setup_dataloaders(dataloader)

        for callback in self._fabric._callbacks:
            if isinstance(callback, DataHooks):
                callback.setup(stage="fit")
            else:
                callback.setup(trainer=self, pl_module=model, stage="fit")

        self._fabric.call("on_train_start", self, model)

        for i in range(self._max_epochs):
            self._epoch_loop(
                model=model,
                optimizers=optimizers,
                dataloader=dataloader,
            )
        self._fabric.call("on_fit_end", trainer=self, pl_module=model)
        self._fabric.call("teardown", trainer=self, pl_module=model)

    def save_checkpoint(self, filepath: Path) -> None:
        raise NotImplementedError(
            "LightlyTrainTrainer does not support saving checkpoints. "
            "Use the Fabric's save_checkpoint method instead."
        )

    def _epoch_loop(
        self,
        model: _FabricModule,
        optimizers: list[_FabricOptimizer],
        dataloader: _FabricDataLoader,
    ) -> None:
        self._fabric.call("on_train_epoch_start", self, model)

        for i, batch in enumerate(dataloader):
            self._training_step(
                model=model, optimizer=optimizers, batch=batch, batch_idx=i
            )

        self._fabric.call("on_train_epoch_end", trainer=self, pl_module=model)

    def _training_step(
        self, model: _FabricModule, optimizers: list[_FabricOptimizer], batch, batch_idx: int
    ) -> None:
        """
        Perform a single training step.
        This method is called by the fit method.
        """
        self._fabric.call(
            "on_train_batch_start",
            trainer=self,
            pl_module=model,
            batch=batch,
            batch_idx=batch_idx,
        )

        for optimizer in optimizers:
            optimizer.zero_grad()        
        loss = model.training_step(batch, batch_idx)
        self._fabric.backward(loss)
        for optimizer in optimizers:
            optimizer.step()

        self._fabric.call(
            "on_train_batch_end",
            trainer=self,
            pl_module=model,
            batch=batch,
            batch_idx=batch_idx,
        )

    @property
    def estimated_stepping_batches(self) -> int:
        return self.max_epochs * self.num_training_batches
    
    @property
    def num_training_batches(self) -> int:
        return self._len_dataloader
