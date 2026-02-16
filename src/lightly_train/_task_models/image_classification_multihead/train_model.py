#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, model_validator

from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.train_model import TrainModelArgs


class ImageClassificationMultiheadTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 128
    default_steps: ClassVar[int] = 100_000

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        TaskSaveCheckpointArgs
    )

    # Backbone args
    backbone_freeze: bool = Field(default=True, frozen=True)

    # Optim
    lr: list[float] | float = 3e-4

    @model_validator(mode="after")
    def _convert_lr_to_list(self) -> ImageClassificationMultiheadTrainArgs:
        """Convert float lr to single-element list."""
        if isinstance(self.lr, float):
            self.lr = [self.lr]
        return self

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
        pass
