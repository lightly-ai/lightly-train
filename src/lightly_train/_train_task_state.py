#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, TypedDict

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from lightly_train.types import TaskBatch


class TrainTaskState(TypedDict):
    train_model: Module
    optimizer: Optimizer
    train_dataloader: DataLoader[TaskBatch]
    step: int
    # Model class path and initialization arguments for serialization.
    # Used to reconstruct the model after training.
    model_class_path: str
    model_init_args: dict[str, Any]
