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
import torchvision.tv_tensors as tv_tensors
from pydantic import Field

from lightly_train._data import file_helpers, label_helpers, yolo_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._transforms.oriented_object_detection_transform import (
    OrientedObjectDetectionCollateFunction,
    OrientedObjectDetectionTransform,
)
from lightly_train._transforms.task_transform import TaskCollateFunction
from lightly_train.types import OrientedObjectDetectionDatasetItem, PathLike


class YOLOOrientedObjectDetectionDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: YOLOOrientedObjectDetectionDatasetArgs  # type: ignore[assignment]

    batch_collate_fn_cls: ClassVar[type[TaskCollateFunction]] = (
        OrientedObjectDetectionCollateFunction
    )

    def __init__(
        self,
        dataset_args: YOLOOrientedObjectDetectionDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: OrientedObjectDetectionTransform,
    ) -> None:
        super().__init__(
            transform=transform, dataset_args=dataset_args, image_info=image_info
        )

        # Get the class mapping.
        self.class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.dataset_args.classes.keys(),
                ignore_classes=self.dataset_args.ignore_classes,
            )
        )

    def __getitem__(self, index: int) -> OrientedObjectDetectionDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])
        label_path = Path(image_info["label_path"]).with_suffix(".txt")

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        image_np = file_helpers.open_image_numpy(image_path)
        h, w, _ = image_np.shape

        if label_path.exists():
            corners_np, class_labels_np = (
                file_helpers.open_yolo_oriented_object_detection_label_numpy(label_path)
            )
        else:
            corners_np = np.zeros((0, 8), dtype=np.float64)
            class_labels_np = np.zeros((0,), dtype=np.int64)

        # Remove instances with class IDs that are not in the included classes.
        keep = np.array(
            [
                int(class_id) in self.class_id_to_internal_class_id
                for class_id in class_labels_np
            ],
            dtype=np.bool_,
        )
        corners_np = corners_np[keep]
        class_labels_np = class_labels_np[keep]

        # Map class IDs to internal class IDs.
        internal_class_labels_np = np.array(
            [
                self.class_id_to_internal_class_id[int(class_id)]
                for class_id in class_labels_np
            ],
            dtype=np.int_,
        )

        # Convert from 4-corner format (x1,y1,x2,y2,x3,y3,x4,y4) to
        # (cx, cy, w, h, angle) format.
        if corners_np.shape[0] > 0:
            bboxes_np = yolo_helpers.oriented_bbox_from_corners(corners_np)
        else:
            bboxes_np = np.zeros((0, 5), dtype=np.float64)

        # Convert numpy arrays to tv_tensors for torchvision transforms.
        # Image: (H, W, C) -> (C, H, W) for tv_tensors.Image
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        tv_image = tv_tensors.Image(image_tensor)

        # Bboxes: normalized (cx, cy, w, h, angle) with canvas size (h, w)
        if bboxes_np.shape[0] > 0:
            tv_bboxes = tv_tensors.BoundingBoxes(  # type: ignore[call-arg]
                torch.from_numpy(bboxes_np),
                format=tv_tensors.BoundingBoxFormat.CXCYWHR,
                canvas_size=(h, w),
            )
        else:
            tv_bboxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 5), dtype=torch.float64),
                format=tv_tensors.BoundingBoxFormat.CXCYWHR,
                canvas_size=(h, w),
            )

        transform_input = {
            "image": tv_image,
            "bboxes": tv_bboxes,
            "class_labels": torch.from_numpy(internal_class_labels_np),
        }

        transformed = self.transform(transform_input)

        image = transformed["image"]
        bboxes = transformed["bboxes"]
        transformed_classes = transformed["class_labels"]

        return OrientedObjectDetectionDatasetItem(
            image_path=str(image_path),
            image=image,
            bboxes=bboxes,
            classes=transformed_classes,
            original_size=(
                w,
                h,
            ),  # TODO (Thomas, 10/25): Switch to (h, w) for consistency.
        )


class YOLOOrientedObjectDetectionDataArgs(TaskDataArgs):
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
    ) -> YOLOOrientedObjectDetectionDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="train",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOOrientedObjectDetectionDatasetArgs(
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
        )

    def get_val_args(self) -> YOLOOrientedObjectDetectionDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="val",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOOrientedObjectDetectionDatasetArgs(
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


class YOLOOrientedObjectDetectionDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    label_dir: Path
    classes: dict[int, str]
    ignore_classes: set[int] | None
    skip_if_label_file_missing: bool

    def list_image_info(self) -> Iterable[dict[str, str]]:
        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / Path(image_filename)
            label_filepath = self.label_dir / Path(image_filename).with_suffix(".txt")

            # TODO (Thomas, 10/25): Log warning if label file does not exist.
            # And keep track of how many files are missing labels.
            if self.skip_if_label_file_missing and not label_filepath.exists():
                continue

            yield {
                "image_path": str(image_filepath),
                "label_path": str(label_filepath),
            }

    @staticmethod
    def get_dataset_cls() -> type[YOLOOrientedObjectDetectionDataset]:
        return YOLOOrientedObjectDetectionDataset
