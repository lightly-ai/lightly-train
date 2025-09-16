#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from torch.utils.data import Dataset

from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import TaskDatasetItem


class TaskDataset(Dataset[TaskDatasetItem]):
    def __init__(self, transform: TaskTransform) -> None:
        self._transform = transform

    @property
    def transform(self) -> TaskTransform:
        return self._transform

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> TaskDatasetItem:
        raise NotImplementedError()
