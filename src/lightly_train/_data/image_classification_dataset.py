#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import Field
from torch import Tensor

from lightly_train._data import file_helpers, label_helpers
from lightly_train._data.task_batch_collation import (
    BaseCollateFunction,
    ImageClassificationCollateFunction,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ImageClassificationDatasetItem, PathLike


class ImageClassificationDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: ImageClassificationDatasetArgs  # type: ignore[assignment]
    batch_collate_fn_cls: ClassVar[type[BaseCollateFunction]] = (
        ImageClassificationCollateFunction
    )

    def __init__(
        self,
        dataset_args: ImageClassificationDatasetArgs,
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
                ignore_classes=self.dataset_args.ignore_classes,
            )
        )

    def parse_and_map_to_internal_class_ids(self, class_ids_str: str) -> Tensor:
        """
        Parse a delimiter-separated string of class IDs and map them to internal class IDs.

        Args:
            class_ids_str: Delimiter-separated class ID string (e.g. "3,7,12").

        Returns:
            1D tensor of internal class IDs (dtype=torch.long).
        """
        class_ids_str = class_ids_str.strip()
        internal_class_ids = []
        for class_id_str in class_ids_str.split(self.dataset_args.label_delimiter):
            class_id_str = class_id_str.strip()
            if class_id_str:
                # Map to internal class id.
                internal_class_id = self.class_id_to_internal_class_id[
                    int(class_id_str)
                ]
                internal_class_ids.append(internal_class_id)
        return torch.tensor(internal_class_ids, dtype=torch.long)

    def __getitem__(self, index: int) -> ImageClassificationDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])
        class_ids_str = image_info["class_id"]

        # Load the image as numpy array.
        image_np = file_helpers.open_image_numpy(image_path)

        # Parse the class ids, remap to internal classes and convert to tensor.
        internal_class_ids = self.parse_and_map_to_internal_class_ids(class_ids_str)

        # Apply the transform to the image.
        transformed = self.transform({"image": image_np})
        image = transformed["image"]

        return ImageClassificationDatasetItem(
            image_path=str(image_path),
            image=image,
            classes=internal_class_ids,
        )


class ImageClassificationDataArgs(TaskDataArgs):
    train: PathLike
    val: PathLike
    test: PathLike | None = None
    classes: dict[int, str]
    label_delimiter: str = ","
    ignore_classes: set[int] | None = Field(default=None, strict=False)

    # Attributes of the .csv files.
    csv_image_column: str = "image_path"
    csv_label_column: str = "label"
    csv_label_type: Literal["name", "id"] = "name"

    classification_task: Literal["multiclass", "multilabel"]

    def train_imgs_path(self) -> Path:
        return Path(self.train)

    def val_imgs_path(self) -> Path:
        return Path(self.val)

    def get_train_args(
        self,
    ) -> ImageClassificationDatasetArgs:
        return ImageClassificationDatasetArgs(
            dir_or_file=Path(self.train),
            classes=self.classes,
            classification_task=self.classification_task,
            csv_image_column=self.csv_image_column,
            csv_label_column=self.csv_label_column,
            csv_label_type=self.csv_label_type,
            label_delimiter=self.label_delimiter,
            ignore_classes=self.ignore_classes,
        )

    def get_val_args(
        self,
    ) -> ImageClassificationDatasetArgs:
        return ImageClassificationDatasetArgs(
            dir_or_file=Path(self.val),
            classes=self.classes,
            classification_task=self.classification_task,
            csv_image_column=self.csv_image_column,
            csv_label_column=self.csv_label_column,
            csv_label_type=self.csv_label_type,
            label_delimiter=self.label_delimiter,
            ignore_classes=self.ignore_classes,
        )

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        ignore_classes = set() if self.ignore_classes is None else self.ignore_classes
        return {
            class_id: class_name
            for class_id, class_name in self.classes.items()
            if class_id not in ignore_classes
        }

    @property
    def num_included_classes(self) -> int:
        return len(self.included_classes)


class ImageClassificationMulticlassDataArgs(ImageClassificationDataArgs):
    classification_task: Literal["multiclass"] = "multiclass"


class ImageClassificationMultilabelDataArgs(ImageClassificationDataArgs):
    classification_task: Literal["multilabel"] = "multilabel"


class ImageClassificationDatasetArgs(TaskDatasetArgs):
    dir_or_file: Path
    classes: dict[int, str]
    ignore_classes: set[int] | None
    classification_task: Literal["multiclass", "multilabel"]

    # CSV columns.
    csv_image_column: str = "image_path"
    csv_label_column: str = "label"

    # Type of the labels in the csv: class names or class ids.
    csv_label_type: Literal["name", "id"] = "name"

    # Delimiter for the labels.
    label_delimiter: str = ","

    def list_image_info(self) -> Iterable[dict[str, str]]:
        if self.dir_or_file.is_dir():
            yield from self._list_image_info_from_folder()
        else:
            yield from self._list_image_info_from_csv()

    def _list_image_info_from_folder(self) -> Iterable[dict[str, str]]:
        # Map directory/class name to class id.
        name_to_id = {name: class_id for class_id, name in self.classes.items()}

        for class_name, class_id in sorted(name_to_id.items(), key=lambda x: x[1]):
            # Don't collect from ignore_classes.
            if self.ignore_classes is not None and class_id in self.ignore_classes:
                continue

            class_dir = self.dir_or_file / class_name
            # Only consider directories that are in `classes`.
            if not class_dir.exists():
                continue
            if not class_dir.is_dir():
                continue

            for image_filename in file_helpers.list_image_filenames_from_dir(
                image_dir=class_dir
            ):
                image_filepath = class_dir / image_filename
                # Labels are comma-separated to support multi-labels.
                yield {
                    "image_path": str(image_filepath),
                    "class_id": str(class_id),
                }

    def _list_image_info_from_csv(self) -> Iterable[dict[str, str]]:
        is_multilabel = self.classification_task == "multilabel"

        # Map directory/class name to class id.
        name_to_id = {name: class_id for class_id, name in self.classes.items()}

        # Verify the .csv files is provided and exists.
        if not self.dir_or_file.exists():
            raise FileNotFoundError(f"CSV file {self.dir_or_file} does not exist.")

        with self.dir_or_file.open("r", newline="") as f:
            reader = csv.DictReader(f)

            # Sanity checks for csv format.
            if reader.fieldnames is None:
                raise ValueError(f"CSV {self.dir_or_file} has no header.")
            if self.csv_image_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV {self.dir_or_file} missing required column '{self.csv_image_column}'. "
                    f"Found columns: {reader.fieldnames}"
                )
            if self.csv_label_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV {self.dir_or_file} missing required column '{self.csv_label_column}'. "
                    f"Found columns: {reader.fieldnames}"
                )

            # Set the directory relative to which paths are resolved.
            root_dir = self.dir_or_file.parent

            # Get the set of supported image extensions.
            supported_image_extensions = file_helpers._supported_image_extensions()

            # Iterate over the csv's rows.
            for row in reader:
                image_path = (row.get(self.csv_image_column) or "").strip()
                labels_str = (row.get(self.csv_label_column) or "").strip()

                # Skip incomplete rows.
                # TODO(Thomas, 01/26): Add a flag to disable skipping invalid rows.
                if image_path == "" or labels_str == "":
                    continue

                # Verify that the file is from a supported image format.
                image_path_p = Path(image_path)
                extension = image_path_p.suffix.lower()
                if extension not in supported_image_extensions:
                    continue

                # Resolve relative paths against the CSV file location.
                if not image_path_p.is_absolute():
                    image_path_p = root_dir / image_path_p
                else:
                    image_path_p = image_path_p

                # Check if the image exists and is a regular file.
                if not image_path_p.is_file():
                    continue

                # Convert to string.
                image_path = str(image_path_p)

                if self.csv_label_type == "name":
                    # Map class names to class IDs.
                    class_ids = {
                        name_to_id[class_name_str.strip()]
                        for class_name_str in labels_str.split(self.label_delimiter)
                        if class_name_str.strip() != ""
                    }
                else:
                    # Handle potential spaces in class IDs.
                    class_ids = {
                        int(class_id.strip())
                        for class_id in labels_str.split(self.label_delimiter)
                        if class_id.strip() != ""
                    }

                # Don't collect from ignore classes.
                if self.ignore_classes is not None:
                    class_ids = {
                        class_id
                        for class_id in class_ids
                        if class_id not in self.ignore_classes
                    }
                    if not class_ids:
                        continue

                if not is_multilabel and len(class_ids) > 1:
                    raise RuntimeError(
                        f"Image '{image_path}' has multiple labels {class_ids} but the "
                        f"classification task is '{self.classification_task}'. Set "
                        "classification_task='multilabel' to enable multilabel "
                        "classification."
                    )

                # Keep only non-ignore classes.
                class_ids_str = self.label_delimiter.join(map(str, class_ids))

                yield {
                    "image_path": image_path,
                    "class_id": class_ids_str,  # can be "3,7,12" (multi-label)
                }

    @staticmethod
    def get_dataset_cls() -> type[ImageClassificationDataset]:
        return ImageClassificationDataset
