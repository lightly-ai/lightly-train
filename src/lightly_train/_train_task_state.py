#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import TypedDict

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from lightly_train.types import TaskBatch


class TrainTaskState(TypedDict):
    model: Module
    optimizer: Optimizer
    train_dataloader: DataLoader[TaskBatch]
    step: int
