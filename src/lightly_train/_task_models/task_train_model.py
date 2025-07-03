#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from lightning_fabric import Fabric
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from lightly_train._configs.config import PydanticConfig


class TaskTrainModelArgs(PydanticConfig):
    pass


class TaskTrainModel(Module):
    """Base class for task-specific models for training. Not exposed to the user.

    This class stores the model, criterion, and metrics for training and validation.
    It also implements the train and validation steps.
    """

    def train_step(self, fabric: Fabric, batch) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        # Forward pass for training step.
        # Return dictionary with loss and metrics for logging.
        raise NotImplementedError()

    def val_step(self, fabric: Fabric, batch) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        # Forward pass for validation step.
        # Return dictionary with loss and metrics for logging.
        raise NotImplementedError()

    def get_optimizer(self) -> Optimizer:
        raise NotImplementedError()
