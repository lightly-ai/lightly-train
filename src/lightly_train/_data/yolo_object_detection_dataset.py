#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pydantic
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ImageFilename, ObjectDetectionDatasetItem


class YoloObjectDetectionDataset(Dataset[ObjectDetectionDatasetItem]):
    def __init__(
        self,
        dataset_args: YoloObjectDetectionDatasetArgs,
        image_filenames: Sequence[ImageFilename],
        transform: TaskTransform,
        mode: Literal["train", "val", "test"],
    ):
        self._args = dataset_args
        self.image_filenames = image_filenames
        self.transform = transform
        self.mode = mode

        self._train_folder_first = self._determine_train_folder_first(train_path=self._args.train, val_path=self._args.val)

    def _get_image_dir(
        self,
        root: Path,
        mode: Literal["train", "val", "test"],
    ) -> Path:
        pass

    def _determine_train_folder_first(
        self,
        train_path: Path,
        val_path: Path,
    ) -> bool:
        # Check if the train folder comes before the val folder in the path.
        return str(train_path).split("/")[-2] < str(val_path).split("/")[-2]

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> ObjectDetectionDatasetItem:
        pass


class YoloObjectDetectionDatasetArgs(PydanticConfig):
    path: Path
    train: Path
    val: Path
    test: Path | None
    names: dict[int, str]

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: Path) -> Path:
        if "images" not in str(v):
            raise ValueError(f"Expected path to include 'images', got {v}")
        return v
