#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._configs.config import PydanticConfig
from lightly_train._data.task_dataset import TaskDatasetArgs


class TaskDataArgs(PydanticConfig):
    @property
    def included_classes(self) -> dict[int, str]:
        raise NotImplementedError()

    def train_data_mmap_hash(self) -> str:
        """Return a str that can be hashed that should identify the train dataset so that image_info
        does not need to be recomputed on restarts.

        Should be fast and not read the whole dataset while still trying to be as accurate as possible.
        """
        raise NotImplementedError()

    def val_data_mmap_hash(self) -> str:
        """Return a str that can be hashed that should identify the val dataset so that image_info
        does not need to be recomputed on restarts.

        Should be fast and not read the whole dataset while still trying to be as accurate as possible.
        """
        raise NotImplementedError()

    def get_train_args(
        self,
    ) -> TaskDatasetArgs:
        raise NotImplementedError()

    def get_val_args(
        self,
    ) -> TaskDatasetArgs:
        raise NotImplementedError()
