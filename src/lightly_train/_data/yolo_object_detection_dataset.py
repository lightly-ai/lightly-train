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
import torch
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ObjectDetectionDatasetItem


class YoloObjectDetectionDatasetArgs(PydanticConfig):
    path: Path
    train: Path
    val: Path
    test: Path | None = None
    names: dict[int, str]

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: Path) -> Path:
        if "images" not in str(v):
            raise ValueError(f"Expected path to include 'images', got {v}.")
        if len(str(v).split("/")) != 2:
            raise ValueError(f"Expected subdirectories of depth 2 from root, got {v}.")
        return v


class YoloObjectDetectionDataset(Dataset[ObjectDetectionDatasetItem]):
    def __init__(
        self,
        dataset_args: YoloObjectDetectionDatasetArgs,
        transform: TaskTransform,
        image_filenames: Sequence[str],
        mode: Literal["train", "val", "test"],
    ) -> None:
        self._args = dataset_args
        self.transform = transform
        self.image_filenames = image_filenames
        self.mode = mode

        self._image_dir, self._label_dir = self._get_image_and_labels_dirs()
        if self._image_dir is None or self._label_dir is None:
            raise ValueError(
                f"Could not find image or label directory for mode '{self.mode}'. "
                "Ensure that the dataset paths are correctly set."
            )

    def _get_image_and_labels_dirs(
        self,
    ) -> tuple[Path | None, Path | None]:
        train_img_dir = self._args.path / self._args.train
        val_img_dir = self._args.path / self._args.val
        test_img_dir = self._args.path / self._args.test if self._args.test else None

        train_label_path = Path(str(self._args.train).replace("images", "labels"))
        val_label_path = Path(str(self._args.val).replace("images", "labels"))
        test_label_path = (
            Path(str(self._args.test).replace("images", "labels"))
            if self._args.test
            else None
        )

        train_label_dir = self._args.path / train_label_path
        val_label_dir = self._args.path / val_label_path
        test_label_dir = self._args.path / test_label_path if test_label_path else None

        if self.mode == "train":
            return train_img_dir, train_label_dir
        elif self.mode == "val":
            return val_img_dir, val_label_dir
        elif self.mode == "test":
            return test_img_dir, test_label_dir
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> ObjectDetectionDatasetItem:
        assert self._image_dir is not None
        assert self._label_dir is not None

        # Load the image.
        image_filename = self.image_filenames[index]
        image_path = self._image_dir / Path(image_filename)
        label_path = self._label_dir / Path(image_filename).with_suffix(".txt")

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

        print(
            type(transformed["image"]),
            type(transformed["bboxes"]),
            type(transformed["class_labels"]),
        )

        image = transformed["image"]
        bboxes = torch.from_numpy(transformed["bboxes"]).float()
        class_labels = torch.from_numpy(transformed["class_labels"]).long()

        return ObjectDetectionDatasetItem(
            image_path=str(image_path),
            image=image,
            bboxes=bboxes,
            classes=class_labels,
        )
