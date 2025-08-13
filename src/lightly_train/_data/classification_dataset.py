#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ImageFilename, PathLike


class ClassificationDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        dataset_args: ClassificationDatasetArgs,
        image_filenames: Sequence[ImageFilename],
        transform: TaskTransform,
    ):
        self.args = dataset_args
        self.image_filenames = image_filenames
        self.transform = transform
        self.classes = dataset_args.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image_filename = self.image_filenames[index]
        image_path = self.args.image_dir / image_filename
        image = file_helpers.open_image_numpy(image_path=image_path, mode="RGB")
        transformed = self.transform({"image": image})
        target = torch.tensor(self.args.targets[image_filename], dtype=torch.long)
        return {
            "image_path": torch.tensor(str(image_path)),  # type: ignore
            "image": transformed["image"],
            "target": target,
        }


class ClassificationDatasetArgs(PydanticConfig):
    image_dir: Path
    classes: list[str]
    targets: dict[str, int]

    @staticmethod
    def get_dataset_cls() -> type[ClassificationDataset]:
        return ClassificationDataset


class SplitArgs(PydanticConfig):
    images: PathLike
    targets: PathLike


class ClassificationDataArgs(TaskDataArgs):
    train: SplitArgs
    val: SplitArgs
    classes: list[str]

    def get_targets(self, targets_path: PathLike) -> dict[str, int]:
        # Assume targets_path is a .npy or .json mapping image filename to class index
        if str(targets_path).endswith(".npy"):
            loaded = np.load(targets_path, allow_pickle=True)
            if isinstance(loaded, dict):
                return dict(loaded)
            return dict(loaded.item())
        elif str(targets_path).endswith(".json"):
            import json

            with open(targets_path, "r") as f:
                loaded = json.load(f)
            # Ensure all values are int
            return {k: int(v) for k, v in loaded.items()}
        else:
            raise ValueError("Unsupported targets file format.")

    def get_train_args(self) -> ClassificationDatasetArgs:
        targets = self.get_targets(self.train.targets)
        return ClassificationDatasetArgs(
            image_dir=Path(self.train.images),
            classes=self.classes,
            targets=targets,
        )

    def get_val_args(self) -> ClassificationDatasetArgs:
        targets = self.get_targets(self.val.targets)
        return ClassificationDatasetArgs(
            image_dir=Path(self.val.images),
            classes=self.classes,
            targets=targets,
        )
