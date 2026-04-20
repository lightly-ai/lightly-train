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
from typing import ClassVar, Sequence

import numpy as np
import pydantic
import torch
from pydantic import Field

from lightly_train._data import file_helpers, label_helpers, yolo_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._env import Env
from lightly_train._transforms.instance_segmentation_transform import (
    InstanceSegmentationCollateFunction,
    InstanceSegmentationTransform,
    InstanceSegmentationTransformArgs,
    InstanceSegmentationTransformInput,
    InstanceSegmentationTransformOutput,
)
from lightly_train._transforms.task_transform import TaskCollateFunction, TaskTransform
from lightly_train.types import (
    BinaryMasksDict,
    InstanceSegmentationDatasetItem,
    PathLike,
)


class InstanceSegmentationDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: YOLOInstanceSegmentationDatasetArgs

    batch_collate_fn_cls: ClassVar[type[TaskCollateFunction]] = (
        InstanceSegmentationCollateFunction
    )

    def __init__(
        self,
        dataset_args: YOLOInstanceSegmentationDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: InstanceSegmentationTransform | None = None,
    ) -> None:
        super().__init__(
            transform=transform, dataset_args=dataset_args, image_info=image_info
        )
        if transform is not None:
            self._init_image_mode(transform)

    def set_transform(self, transform: TaskTransform) -> None:
        super().set_transform(transform)
        assert isinstance(transform, InstanceSegmentationTransform)
        self._init_image_mode(transform)

    def _init_image_mode(self, transform: InstanceSegmentationTransform) -> None:
        transform_args = transform.transform_args
        assert isinstance(transform_args, InstanceSegmentationTransformArgs)

        image_mode = (
            None
            if Env.LIGHTLY_TRAIN_IMAGE_MODE.value is None
            else ImageMode(Env.LIGHTLY_TRAIN_IMAGE_MODE.value)
        )
        if image_mode is None:
            image_mode = (
                ImageMode.RGB
                if transform_args.num_channels == 3
                else ImageMode.UNCHANGED
            )

        if image_mode not in (ImageMode.RGB, ImageMode.UNCHANGED):
            raise ValueError(
                f"Invalid image mode: '{image_mode}'. "
                f"Supported modes are '{[ImageMode.RGB.value, ImageMode.UNCHANGED.value]}'."
            )
        self.image_mode = image_mode

    def __getitem__(self, index: int) -> InstanceSegmentationDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])
        polygons = json.loads(image_info["polygons"])
        bboxes = json.loads(image_info["bboxes"])
        class_labels = json.loads(image_info["class_labels"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

        image_np = file_helpers.open_image_numpy(image_path)
        polygons_np = [np.array(polygon, dtype=np.float64) for polygon in polygons]
        bboxes_np = np.array(bboxes, dtype=np.float64).reshape(len(bboxes), 4)
        class_labels_np = np.array(class_labels, dtype=np.int64)

        binary_masks_np = yolo_helpers.binary_masks_from_polygons(
            polygons=polygons_np, height=image_np.shape[0], width=image_np.shape[1]
        )

        transform_input: InstanceSegmentationTransformInput = {
            "image": image_np,
            # Shape (n_instances, H, W)
            "binary_masks": binary_masks_np.astype(np.uint8),
            "bboxes": bboxes_np,  # Shape (n_instances, 4)
            "class_labels": class_labels_np,  # Shape (n_instances,)
        }

        transformed: InstanceSegmentationTransformOutput = self.transform(
            transform_input
        )

        image = transformed["image"]
        # Some albumentations versions return lists of tuples instead of arrays.
        if isinstance(transformed["bboxes"], list):
            transformed["bboxes"] = np.array(transformed["bboxes"])
        if isinstance(transformed["class_labels"], list):
            transformed["class_labels"] = np.array(transformed["class_labels"])

        bboxes = torch.from_numpy(transformed["bboxes"]).float()
        class_labels = torch.from_numpy(transformed["class_labels"]).long()
        # Match format from MaskSemanticSegmentationDatasetItem
        binary_masks: BinaryMasksDict = {
            "masks": transformed["binary_masks"].bool(),
            "labels": class_labels,
        }

        return InstanceSegmentationDatasetItem(
            image_path=str(image_path),
            image=image,
            binary_masks=binary_masks,
            bboxes=bboxes,
            classes=class_labels,
        )


class YOLOInstanceSegmentationDataArgs(TaskDataArgs):
    ignore_index: ClassVar[int | None] = None
    path: PathLike
    train: PathLike
    val: PathLike
    # TODO(Guarin, 10/25): Handle test set.
    test: PathLike | None = None
    # "names" instead of "classes" to match YOLO convention.
    names: dict[int, str]
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False

    def train_imgs_path(self) -> Path:
        return Path(self.path) / self.train

    def val_imgs_path(self) -> Path:
        return Path(self.path) / self.val

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: PathLike) -> Path:
        v = Path(v)
        if "images" not in v.parts:
            raise ValueError(f"Expected path to include 'images' directory, got {v}.")
        return v

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        ignore_classes = set() if self.ignore_classes is None else self.ignore_classes
        return {
            class_id: class_name
            for class_id, class_name in self.names.items()
            if class_id not in ignore_classes
        }

    @property
    def num_included_classes(self) -> int:
        return len(self.included_classes)

    def get_train_args(
        self,
    ) -> YOLOInstanceSegmentationDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="train",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOInstanceSegmentationDatasetArgs(
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
        )

    def get_val_args(self) -> YOLOInstanceSegmentationDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="val",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOInstanceSegmentationDatasetArgs(
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
        )


class YOLOInstanceSegmentationDatasetArgs(TaskDatasetArgs):
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
            image_filepath = self.image_dir / Path(image_filename)
            label_filepath = self.label_dir / Path(image_filename).with_suffix(".txt")

            if label_filepath.exists():
                polygons, bboxes, class_labels = (
                    file_helpers.open_yolo_instance_segmentation_label(
                        label_path=label_filepath
                    )
                )
            else:
                # TODO (Thomas, 10/25): Log warning if label file does not exist.
                #   And keep track of how many files are missing labels.
                if self.skip_if_label_file_missing:
                    continue
                polygons = []
                bboxes = []
                class_labels = []

            # Remove instances with class IDs that are not in the included classes
            keep = [label in class_id_to_internal_class_id for label in class_labels]
            polygons = list(itertools.compress(polygons, keep))
            bboxes = list(itertools.compress(bboxes, keep))
            class_labels = list(itertools.compress(class_labels, keep))

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield {
                "image_path": str(image_filepath),
                "polygons": json.dumps(polygons),
                "bboxes": json.dumps(bboxes),
                "class_labels": json.dumps(class_labels),
            }

    @staticmethod
    def get_dataset_cls() -> type[InstanceSegmentationDataset]:
        return InstanceSegmentationDataset
