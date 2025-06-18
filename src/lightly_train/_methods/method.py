#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from lightly.utils.scheduler import CosineWarmupScheduler
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger as LightningWandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

import lightly_train._plot as methods_helpers
from lightly_train._loggers.mlflow import MLFlowLogger
from lightly_train._loggers.tensorboard import TensorBoardLogger
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim import optimizer_helpers
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch


@dataclass
class TrainingStepResult:
    loss: Tensor
    log_dict: Mapping[str, Any] | None = None


@dataclass
class BatchStartEndTime:
    batch_start_s: float | None = None
    batch_end_s: float | None = None


class Method(LightningModule):
    def __init__(
        self,
        method_args: MethodArgs,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
    ):
        super().__init__()
        self.global_batch_size = global_batch_size
        self.optimizer_args: OptimizerArgs = optimizer_args
        self.batch_start_end_time = BatchStartEndTime()

    @staticmethod
    def method_args_cls() -> type[MethodArgs]:
        """Return the class of the method args.

        Overwrite this method to change the class of the method args.
        """
        raise NotImplementedError()

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        """Return the optimizer args class.

        Overwrite this method to change the optimizer args class.
        """
        optim_type = AdamWArgs.type() if optim_type == "auto" else optim_type
        return optimizer_helpers.get_optimizer_args_cls(optim_type=optim_type)

    def trainable_modules(self) -> TrainableModules:
        """Return the modules that should be optimized."""
        raise NotImplementedError()

    # Ignore the return type, because pytorch-lightning types it wrongly.
    # See https://github.com/Lightning-AI/pytorch-lightning/issues/20106
    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Scale the learning rate based on the global batch size.
        lr_scale: float = self.global_batch_size / self.method_args.reference_batch_size
        if self.method_args.lr_scale_method == "sqrt":
            lr_scale = math.sqrt(lr_scale)

        optim = optimizer_helpers.get_optimizer(
            optim_args=self.optimizer_args,
            trainable_modules=self.trainable_modules(),
            lr_scale=lr_scale,
        )

        if self.trainer.max_epochs is None:
            raise RuntimeError("Max epochs is not set.")

        max_epochs = max(1, self.trainer.max_epochs)

        # Warmup for 10 epochs or 10% of the total number of epochs if max_epochs < 100
        warmup_epochs = min(10, max_epochs / 10)
        warmup_steps = min(
            int(self.trainer.estimated_stepping_batches),
            int(self.trainer.estimated_stepping_batches / max_epochs * warmup_epochs),
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optim,
                # The arguments are called "epochs" but they can also be steps.
                warmup_epochs=warmup_steps,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optim], [scheduler]  # type: ignore[return-value]

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        """Return the default transform.

        Overwrite this method to change the transform.
        """
        raise NotImplementedError()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        training_step_log = self.training_step_impl(batch, batch_idx)
        loss = training_step_log.loss
        views = batch["views"]
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(views[0])
        )
        if training_step_log.log_dict is not None:
            self.log_dict(
                training_step_log.log_dict,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(views[0]),
            )
        if self.global_step == 0:
            # Show example views of the images in the first batch only.
            self._log_example_views(train_batch=batch)
        return loss

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        """Execute a training step without logging.

        Overwrite this method to change the training step implementation.
        """
        raise NotImplementedError()

    def on_train_batch_start(self, batch: Batch, batch_idx: int) -> None:
        self._log_time_batch_start()

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        self._log_time_batch_end()

    def _log_example_views(self, train_batch: Batch) -> None:
        example_aug_pil = methods_helpers.plot_example_augmentations(
            train_batch=train_batch
        )
        example_aug_tensor = torchvision_functional.pil_to_tensor(example_aug_pil)

        for logger in self.loggers:
            # TODO(Malte, 09/2024): Replace the instance checks with an abstraction.
            # TODO(Malte, 09/2024): Add an image saving method to the JSONLogger and use it.
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    tag="augmentations",
                    img_tensor=example_aug_tensor,
                    global_step=self.global_step,
                )
            elif isinstance(logger, LightningWandbLogger) or isinstance(
                logger, MLFlowLogger
            ):
                logger.log_image(
                    key="augmentations",
                    images=[example_aug_pil],
                    step=self.global_step,
                )

    def _log_time_batch_start(self) -> None:
        self.batch_start_end_time.batch_start_s = time.perf_counter()
        if self.batch_start_end_time.batch_end_s is not None:
            assert (
                self.batch_start_end_time.batch_start_s
                > self.batch_start_end_time.batch_end_s
            )
            self.log(
                "profiling/data_time",
                self.batch_start_end_time.batch_start_s
                - self.batch_start_end_time.batch_end_s,
            )

    def _log_time_batch_end(self) -> None:
        self.batch_start_end_time.batch_end_s = time.perf_counter()
        if self.batch_start_end_time.batch_start_s is not None:
            assert (
                self.batch_start_end_time.batch_end_s
                > self.batch_start_end_time.batch_start_s
            )
            self.log(
                "profiling/batch_time",
                self.batch_start_end_time.batch_end_s
                - self.batch_start_end_time.batch_start_s,
            )
