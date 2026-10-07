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
from lightly_train._data.item_store import ItemStore, as_item_store
from lightly_train._transforms.task_transform import TaskCollateFunction, TaskTransform
from lightly_train.types import TaskDatasetItem


class TaskDatasetArgs(PydanticConfig):
    def list_image_info(self) -> Iterable[dict[str, str]]:
        """Listing the image info should not happen in-memory for large datasets."""
        raise NotImplementedError()

    def get_dataset_cls(self) -> type[TaskDataset]:
        raise NotImplementedError()


class TaskDataset(Dataset[TaskDatasetItem]):
    batch_collate_fn_cls: ClassVar[type[TaskCollateFunction]] = TaskCollateFunction

    def __init__(
        self,
        dataset_args: TaskDatasetArgs,
        image_info: ItemStore | Sequence[dict[str, str]],
        transform: TaskTransform | None = None,
    ) -> None:
        self.dataset_args = dataset_args
        # Datasets access the rows through the item store interface only. This keeps
        # them independent of how the rows are stored.
        self.image_info: ItemStore = as_item_store(image_info)
        self._transform = transform

    @property
    def transform(self) -> TaskTransform:
        if self._transform is None:
            raise RuntimeError(
                "Transform has not been set on the dataset. Call "
                "`dataset.set_transform(transform)` before accessing it."
            )
        return self._transform

    def set_transform(self, transform: TaskTransform) -> None:
        self._transform = transform

    def get_batch_collate_fn_cls(self) -> type[TaskCollateFunction]:
        return self.batch_collate_fn_cls

    def __len__(self) -> int:
        return len(self.image_info)

    def __getitem__(self, index: int) -> TaskDatasetItem:
        raise NotImplementedError()
