#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from lightning_fabric import Fabric
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models.task_model import TaskModel
from lightly_train._transforms.task_transform import TaskTransform


class TrainModelArgs(PydanticConfig):
    # Default values that are used if the user didn't specify the value manually.
    # We store this in the model args as these values depend on the model. The values
    # are ClassVar because they have to be accessed before the class is instantiated.
    default_batch_size: ClassVar[int]
    default_steps: ClassVar[int]

    def resolve_auto(self, total_steps: int, model_name: str) -> None:
        pass


class TrainModel(Module):
    """Base class for task-specific models for training. Not exposed to the user.

    This class stores the model, criterion, and metrics for training and validation.
    It also implements the train and validation steps.
    """

    task: ClassVar[str]
    train_model_args_cls: ClassVar[type[TrainModelArgs]]
    task_model_cls: ClassVar[type[TaskModel]]
    train_transform_cls: ClassVar[type[TaskTransform]]
    val_transform_cls: ClassVar[type[TaskTransform]]

    # NOTE(Guarin, 07/25): We use the same method names as for LightningModule as
    # those methods are automatically handled by Fabric. Methods with different
    # names that are called within a Fabric context will raise an error if they have
    # not been registered with Fabric.
    # See https://lightning.ai/docs/fabric/2.5.2/api/wrappers.html#using-methods-other-than-forward-for-computation
    # The following methods are automatically handled:
    # - forward
    # - training_step
    # - validation_step
    # - test_step
    # - predict_step
    # See: https://github.com/Lightning-AI/pytorch-lightning/blob/95f16c12fe23664ffa5198a43266f715717c6f45/src/lightning/fabric/wrappers.py#L47-L48

    def training_step(self, fabric: Fabric, batch, step: int) -> TaskStepResult:  # type: ignore[no-untyped-def]
        # Forward pass for training step.
        # Return dictionary with loss and metrics for logging.
        raise NotImplementedError()

    def validation_step(self, fabric: Fabric, batch) -> TaskStepResult:  # type: ignore[no-untyped-def]
        # Forward pass for validation step.
        # Return dictionary with loss and metrics for logging.
        raise NotImplementedError()

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        raise NotImplementedError()

    def set_train_mode(self) -> None:
        """Set the model to training mode."""
        self.train()

    def get_task_model(self) -> TaskModel:
        """Returns the task model.

        This is the model that users interact with for inference and deployment.
        """
        raise NotImplementedError()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        pass


@dataclass
class TaskStepResult:
    loss: Tensor
    log_dict: dict[str, Any]
