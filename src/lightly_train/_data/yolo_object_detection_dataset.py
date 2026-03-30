#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pydantic
from pydantic import Field

from lightly_train._data import file_helpers, label_helpers, yolo_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionCollateFunction,
)
from lightly_train._transforms.task_transform import TaskCollateFunction
from lightly_train.types import ObjectDetectionDatasetItem, PathLike


class YOLOObjectDetectionDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: YOLOObjectDetectionDatasetArgs  # type: ignore[assignment]

    batch_collate_fn_cls: ClassVar[type[TaskCollateFunction]] = (
        ObjectDetectionCollateFunction
    )

    def __getitem__(self, index: int) -> ObjectDetectionDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        image_np = file_helpers.open_image_numpy(image_path)
        h, w, _ = image_np.shape

        bboxes = json.loads(image_info["bboxes"])
        class_labels = json.loads(image_info["class_labels"])
        # TODO (simon, 03/26) do we need this assert?
        assert len(bboxes) == len(class_labels)

        bboxes_np = np.array(bboxes, dtype=np.float64).reshape(len(bboxes), 4)
        class_labels_np = np.array(class_labels, dtype=np.int64)

        transformed = self.transform(
            {
                "image": image_np,
                "bboxes": bboxes_np,  # Shape (n_boxes, 4)
                "class_labels": class_labels_np,  # Shape (n_boxes,)
            }
        )

        return ObjectDetectionDatasetItem(
            image_path=str(image_path),
            image=transformed["image"],
            bboxes=transformed["bboxes"],
            classes=transformed["class_labels"],
            # TODO (Thomas, 10/25): Switch to (h, w) for consistency.
            original_size=(w, h),
        )


class YOLOObjectDetectionDataArgs(TaskDataArgs):
    # TODO: (Lionel, 08/25): Handle test set.
    path: PathLike
    train: PathLike
    val: PathLike
    test: PathLike | None = None
    names: dict[int, str]
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False

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
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
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
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
        )

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        ignore_classes = set() if self.ignore_classes is None else self.ignore_classes
        return {
            class_id: class_name
            for class_id, class_name in self.names.items()
            if class_id not in ignore_classes
        }


class YOLOObjectDetectionDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    label_dir: Path
    classes: dict[int, str]
    ignore_classes: set[int] | None
    skip_if_label_file_missing: bool

    def list_image_info(self) -> Iterable[dict[str, str]]:

        class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.classes.keys(),
                ignore_classes=self.ignore_classes,
            )
        )

        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / image_filename
            label_filepath = (self.label_dir / image_filename).with_suffix(".txt")

            if label_filepath.exists():
                bboxes, class_labels = file_helpers.open_yolo_object_detection_label(
                    label_filepath
                )
            else:
                # TODO (Thomas, 10/25): Log warning if label file does not exist.
                #   And keep track of how many files are missing labels.
                if self.skip_if_label_file_missing:
                    continue
                bboxes = []
                class_labels = []

            # Remove instances with class IDs that are not in the included classes
            keep = [label in class_id_to_internal_class_id for label in class_labels]
            bboxes = list(itertools.compress(bboxes, keep))
            class_labels = list(itertools.compress(class_labels, keep))

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield {
                "image_path": str(image_filepath),
                "bboxes": json.dumps(bboxes),
                "class_labels": json.dumps(class_labels),
            }

    @staticmethod
    def get_dataset_cls() -> type[YOLOObjectDetectionDataset]:
        return YOLOObjectDetectionDataset
