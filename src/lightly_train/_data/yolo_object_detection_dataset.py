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

import numpy as np
import pydantic
import torch
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ImageFilename, ObjectDetectionDatasetItem, PathLike


class YOLOObjectDetectionDataset(Dataset[ObjectDetectionDatasetItem]):
    def __init__(
        self,
        dataset_args: YOLOObjectDetectionDatasetArgs,
        image_filenames: Sequence[ImageFilename],
        transform: TaskTransform,
    ) -> None:
        self.args = dataset_args
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> ObjectDetectionDatasetItem:
        # Load the image.
        image_filename = self.image_filenames[index]
        image_path = self.args.image_dir / Path(image_filename)
        label_path = self.args.label_dir / Path(image_filename).with_suffix(".txt")

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file {label_path} does not exist.")

        image_ = file_helpers.open_image_numpy(image_path)
        bboxes_, class_labels_ = file_helpers.open_yolo_label_numpy(label_path)

        transformed = self.transform(
            {
                "image": image_,
                "bboxes": bboxes_,  # Shape (n_boxes, 4)
                "class_labels": class_labels_,  # Shape (n_boxes,)
            }
        )

        image = transformed["image"]
        # Some albumentations versions return lists of tuples instead of arrays.
        if isinstance(transformed["bboxes"], list):
            transformed["bboxes"] = np.array(transformed["bboxes"])
        if isinstance(transformed["class_labels"], list):
            transformed["class_labels"] = np.array(transformed["class_labels"])
        bboxes = torch.from_numpy(transformed["bboxes"]).float()
        class_labels = torch.from_numpy(transformed["class_labels"]).long()

        return ObjectDetectionDatasetItem(
            image_path=str(image_path),
            image=image,
            bboxes=bboxes,
            classes=class_labels,
        )


class YOLOObjectDetectionDataArgs(TaskDataArgs):
    # TODO: (Lionel, 08/25): Handle test set.
    path: PathLike
    train: PathLike
    val: PathLike
    test: PathLike | None = None
    names: dict[int, str]

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: PathLike) -> Path:
        v = Path(v)
        if "images" not in v.parts:
            raise ValueError(f"Expected path to include 'images' directory, got {v}.")
        return v

    def get_train_args(
        self,
    ) -> YOLOObjectDetectionDatasetArgs:
        image_dir, label_dir = self._get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="train",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOObjectDetectionDatasetArgs(
            image_dir=image_dir, label_dir=label_dir, classes=self.names
        )

    def get_val_args(self) -> YOLOObjectDetectionDatasetArgs:
        image_dir, label_dir = self._get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="val",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOObjectDetectionDatasetArgs(
            image_dir=image_dir, label_dir=label_dir, classes=self.names
        )

    def _get_image_and_labels_dirs(
        self,
        path: Path,
        train: Path,
        val: Path,
        test: Path | None,
        mode: Literal["train", "val", "test"],
    ) -> tuple[Path | None, Path | None]:
        train_img_dir = path / train
        val_img_dir = path / val
        test_img_dir = path / test if test else None

        def _replace_first_images_with_labels(path: Path) -> Path:
            """Replaces only the first occurrence of 'images' with 'labels' in a Path."""
            parts = list(path.parts)
            for i, part in enumerate(parts):
                if part == "images":
                    parts[i] = "labels"
                    break
            return Path(*parts)

        train_label_path = _replace_first_images_with_labels(train)
        val_label_path = _replace_first_images_with_labels(val)
        test_label_path = _replace_first_images_with_labels(test) if test else None

        train_label_dir = path / train_label_path
        val_label_dir = path / val_label_path
        test_label_dir = path / test_label_path if test_label_path else None

        if mode == "train":
            return train_img_dir, train_label_dir
        elif mode == "val":
            return val_img_dir, val_label_dir
        elif mode == "test":
            return test_img_dir, test_label_dir
        else:
            raise ValueError(f"Unknown mode: {mode}")


class YOLOObjectDetectionDatasetArgs(PydanticConfig):
    image_dir: Path
    label_dir: Path
    classes: dict[int, str]
