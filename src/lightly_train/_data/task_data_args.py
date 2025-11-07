#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from lightly_train._configs.config import PydanticConfig
from lightly_train._data.task_dataset import TaskDatasetArgs


class TaskDataArgs(PydanticConfig):
    @property
    def included_classes(self) -> dict[int, str]:
        raise NotImplementedError()

    def train_imgs_path(self) -> Path:
        raise NotImplementedError()

    def val_imgs_path(self) -> Path:
        raise NotImplementedError()

    def get_train_args(
        self,
    ) -> TaskDatasetArgs:
        raise NotImplementedError()

    def get_val_args(
        self,
    ) -> TaskDatasetArgs:
        raise NotImplementedError()
