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

import torch
from torch import Tensor

from lightly_train._data import file_helpers, label_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import ImageClassificationDatasetItem, PathLike


class ImageClassificationDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: ImageClassificationDatasetArgs  # type: ignore[assignment]

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
                ignore_classes=None,
            )
        )

    def parse_and_map_to_internal_class_ids(self, class_ids_str: str) -> Tensor:
        """
        Parse a comma-separated string of class IDs and map them to internal class IDs.

        Args:
            class_ids_str: Comma-separated class ID string (e.g. "3,7,12").

        Returns:
            1D tensor of internal class IDs (dtype=torch.long).
        """
        class_ids_str = class_ids_str.strip()
        internal_class_ids = []
        for class_id_str in class_ids_str.split(","):
            if class_id_str.strip() != "":
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

        # Verify that the image exists.
        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        # Load the image as numpy array.
        image_np = file_helpers.open_image_numpy(image_path)

        # Parse the class ids, remap to internal classes and convert to tensor
        internal_class_ids = self.parse_and_map_to_internal_class_ids(class_ids_str)

        # Apply the transform to the image
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
    names: dict[int, str]

    def train_imgs_path(self) -> Path:
        return Path(self.train)

    def val_imgs_path(self) -> Path:
        return Path(self.val)

    def get_train_args(
        self,
    ) -> ImageClassificationDatasetArgs:
        return ImageClassificationDatasetArgs(
            image_dir=Path(self.train),
            classes=self.names,
        )

    def get_val_args(
        self,
    ) -> ImageClassificationDatasetArgs:
        return ImageClassificationDatasetArgs(
            image_dir=Path(self.val),
            classes=self.names,
        )

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        # TODO(Thomas, 01/26): Implement ignore classes.
        return self.names


class ImageClassificationDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    classes: dict[int, str]

    def list_image_info(self) -> Iterable[dict[str, str]]:
        # Map directory/class name to class id.
        name_to_id = {name: class_id for class_id, name in self.classes.items()}

        for class_name, class_id in sorted(name_to_id.items(), key=lambda x: x[1]):
            class_dir = self.image_dir / class_name
            # Only consider directories that are in `classes`.
            if not class_dir.exists():
                continue
            if not class_dir.is_dir():
                continue

            for image_filename in file_helpers.list_image_filenames_from_dir(
                image_dir=class_dir
            ):
                image_filepath = class_dir / Path(image_filename)
                # Labels are comma-separated to support multi-labels.
                yield {
                    "image_path": str(image_filepath),
                    "class_id": ",".join(map(str, [class_id])),
                }

    @staticmethod
    def get_dataset_cls() -> type[ImageClassificationDataset]:
        return ImageClassificationDataset
