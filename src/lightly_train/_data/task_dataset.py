#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar

from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data.task_batch_collation import BaseCollateFunction
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import TaskDatasetItem


class TaskDatasetArgs(PydanticConfig):
    def list_image_info(self) -> Iterable[dict[str, str]]:
        """Listing the image info should not happen in-memory for large datasets."""
        raise NotImplementedError()

    def get_dataset_cls(self) -> type[TaskDataset]:
        raise NotImplementedError()


class TaskDataset(Dataset[TaskDatasetItem]):
    batch_collate_fn_cls: ClassVar[type[BaseCollateFunction]] = BaseCollateFunction

    def __init__(
        self,
        dataset_args: TaskDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: TaskTransform,
    ) -> None:
        self.dataset_args = dataset_args
        self.image_info = image_info
        self._transform = transform

    @property
    def transform(self) -> TaskTransform:
        return self._transform

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, index: int) -> TaskDatasetItem:
        raise NotImplementedError()
