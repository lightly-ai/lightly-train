#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any

from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike


class DINOv2Classification(TaskModel):
    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        class_ignore_index: int | None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(locals(), ignore_args={"backbone_weights"})
