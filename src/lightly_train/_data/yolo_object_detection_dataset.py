#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pydantic
import torch

from lightly_train._data import file_helpers, label_helpers, yolo_helpers
from lightly_train._data.task_batch_collation import (
    BaseCollateFunction,
    ObjectDetectionCollateFunction,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ObjectDetectionDatasetItem, PathLike


class YOLOObjectDetectionDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: YOLOObjectDetectionDatasetArgs  # type: ignore[assignment]

    batch_collate_fn_cls: ClassVar[type[BaseCollateFunction]] = (
        ObjectDetectionCollateFunction
    )

    def __init__(
        self,
        dataset_args: YOLOObjectDetectionDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: TaskTransform,
    ) -> None:
        super().__init__(
            transform=transform, dataset_args=dataset_args, image_info=image_info
        )

        # Get the class mapping.
        self.class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.dataset_args.classes.keys(),
                ignore_classes=None,
            )
        )

    def __getitem__(self, index: int) -> ObjectDetectionDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])
        label_path = Path(image_info["label_path"]).with_suffix(".txt")

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file {label_path} does not exist.")

        image_np = file_helpers.open_image_numpy(image_path)
        h, w, _ = image_np.shape
        bboxes_np, class_labels_np = (
            file_helpers.open_yolo_object_detection_label_numpy(label_path)
        )

        # Map class IDs to internal class IDs.
        internal_class_labels_np = np.array(
            [
                self.class_id_to_internal_class_id[int(class_id)]
                for class_id in class_labels_np
            ]
        )

        transformed = self.transform(
            {
                "image": image_np,
                "bboxes": bboxes_np,  # Shape (n_boxes, 4)
                "class_labels": internal_class_labels_np,  # Shape (n_boxes,)
            }
        )

        image = transformed["image"]
        # Some albumentations versions return lists of tuples instead of arrays.
        if isinstance(transformed["bboxes"], list):
            transformed["bboxes"] = np.array(transformed["bboxes"])
        if isinstance(transformed["class_labels"], list):
            transformed["class_labels"] = np.array(transformed["class_labels"])
        bboxes = torch.from_numpy(transformed["bboxes"]).float()
        internal_class_labels = torch.from_numpy(transformed["class_labels"]).long()

        return ObjectDetectionDatasetItem(
            image_path=str(image_path),
            image=image,
            bboxes=bboxes,
            classes=internal_class_labels,
            original_size=(
                w,
                h,
            ),  # TODO (Thomas, 10/25): Switch to (h, w) for consistency.
        )


class YOLOObjectDetectionDataArgs(TaskDataArgs):
    # TODO: (Lionel, 08/25): Handle test set.
    path: PathLike
    train: PathLike
    val: PathLike
    test: PathLike | None = None
    names: dict[int, str]

    def train_imgs_path(self) -> Path:
        return Path(self.train)

    def val_imgs_path(self) -> Path:
        return Path(self.val)

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: PathLike) -> Path:
        v = Path(v)
        if "images" not in v.parts:
            raise ValueError(f"Expected path to include 'images' directory, got {v}.")
        return v

    def get_train_args(
        self,
    ) -> YOLOObjectDetectionDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
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
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
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

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        return self.names


class YOLOObjectDetectionDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    label_dir: Path
    classes: dict[int, str]

    def list_image_info(self) -> Iterable[dict[str, str]]:
        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / Path(image_filename)
            label_filepath = self.label_dir / Path(image_filename).with_suffix(".txt")
            # TODO (Thomas, 10/25): Log warning if label file does not exist.
            # And keep track of how many files are missing labels.
            if label_filepath.exists():
                yield {
                    "image_path": str(image_filepath),
                    "label_path": str(label_filepath),
                }

    @staticmethod
    def get_dataset_cls() -> type[YOLOObjectDetectionDataset]:
        return YOLOObjectDetectionDataset
