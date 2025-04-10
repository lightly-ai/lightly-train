#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from torch.nn import Flatten

from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._methods.simclr.simclr_transform import (
    SimCLRTransform,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.sgd_args import SGDArgs
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch


class SimCLRArgs(MethodArgs):
    """Args for SimCLR method."""

    hidden_dim: int = 2048
    output_dim: int = 128
    num_layers: int = 2
    batch_norm: bool = True
    temperature: float = 0.1


class SimCLRSGDArgs(SGDArgs):
    lr: float = 0.3


class SimCLR(Method):
    def __init__(
        self,
        method_args: SimCLRArgs,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
    ):
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
        )
        self.method_args = method_args
        self.embedding_model = embedding_model
        self.flatten = Flatten(start_dim=1)
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.embedding_model.embed_dim,
            hidden_dim=self.method_args.hidden_dim,
            output_dim=self.method_args.output_dim,
            num_layers=self.method_args.num_layers,
            batch_norm=self.method_args.batch_norm,
        )
        self.criterion = NTXentLoss(
            temperature=self.method_args.temperature,
            gather_distributed=dist.is_available(),
        )

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        views = batch["views"]
        x = self.embedding_model(torch.cat(views))
        x = self.flatten(x)
        x = self.projection_head(x)
        x0, x1 = x.chunk(len(views))
        loss = self.criterion(x0, x1)

        return TrainingStepResult(loss=loss)

    @staticmethod
    def method_args_cls() -> type[SimCLRArgs]:
        return SimCLRArgs

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": SimCLRSGDArgs,
            OptimizerType.SGD: SimCLRSGDArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(modules=[self.embedding_model, self.projection_head])

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        return SimCLRTransform
