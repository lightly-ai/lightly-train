#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import (
    ImageFilename,
    MaskSemanticSegmentationDatasetItem,
    PathLike,
)

logger = logging.getLogger(__name__)


class MaskSemanticSegmentationDataset(Dataset[MaskSemanticSegmentationDatasetItem]):
    def __init__(
        self,
        dataset_args: MaskSemanticSegmentationDatasetArgs,
        image_filenames: Sequence[ImageFilename],
        transform: TaskTransform,
    ):
        self.args = dataset_args
        self.image_filenames = image_filenames
        self.transform = transform

        # Get the class mappings.
        self.class_mappings = self.get_class_mappings()
        self.valid_classes = np.array(list(self.class_mappings.keys()))

        # Optionally filter image filenames corresponding to empty targets.
        if dataset_args.check_empty_targets:
            self.filter_empty_targets()

    def is_mask_valid(self, mask: NDArray[np.uint8]) -> bool:
        # Get unique values in the mask.
        unique_values = np.unique(mask)

        # Uniform masks are discarded.
        if len(unique_values) == 1:
            return False

        # Check if at least one value in the mask is in the valid classes.
        return bool(np.isin(unique_values, self.valid_classes).any())

    def filter_empty_targets(self) -> None:
        # Instantiate new list of file names
        new_image_filenames = []

        # Populate the new lists with file names corresponding to valid targets.
        for filename in self.image_filenames:
            filepath = (self.args.mask_dir / filename).with_suffix(".png")

            mask = file_helpers.open_image(image_path=filepath, mode="MASK")
            if self.is_mask_valid(mask):
                new_image_filenames.append(filename)

        # Display the number of filtered files.
        n_filtered_files = len(self) - len(new_image_filenames)
        logger.info(f"Filtered {n_filtered_files} invalid masks out of {len(self)}.")

        # Update the list of valid files.
        self.image_filenames = new_image_filenames

    def get_class_mappings(self) -> dict[int, int]:
        # Verify the classes are set (for mypy).
        assert self.args.classes is not None, (
            "Segmentation dataset classes must be set."
        )

        # Set variables.
        original_classes = self.args.classes.keys()
        ignore_classes = self.args.ignore_classes
        class_mappings = {}
        class_counter = 0

        # Iterate over the classes and populate the class_mapppings.
        for original_class in original_classes:
            if original_class not in ignore_classes:
                # Re-map the class.
                class_mappings[original_class] = class_counter

                # Update the class counter.
                class_counter += 1
        return class_mappings

    def __len__(self) -> int:
        return len(self.image_filenames)

    def get_binary_masks(self, mask: Tensor) -> dict[str, Tensor]:
        # This follows logic from:
        # https://github.com/tue-mps/eomt/blob/716cbd562366b9746804579b48b866da487d9485/datasets/ade20k_semantic.py#L47-L48

        img_masks = []
        img_labels = []
        class_ids = mask.unique().tolist()  # type: ignore[no-untyped-call]

        # Iterate over the labels present in the mask.
        for class_id in class_ids:
            # Check if the class id is the valid classes.
            if class_id not in self.valid_classes:
                continue

            # Create binary mask for the class.
            img_masks.append(mask == class_id)

            # Store the class label.
            img_labels.append(self.class_mappings[class_id])

        # Store the targets.
        targets = {
            "masks": torch.stack(img_masks),
            "labels": mask.new_tensor(img_labels, dtype=torch.long),
        }
        return targets

    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        image_filename = self.image_filenames[index]
        image_path = self.args.image_dir / image_filename
        mask_path = (self.args.mask_dir / image_filename).with_suffix(".png")

        # Load the image and the mask.
        image = file_helpers.open_image(image_path=image_path, mode="RGB")
        mask = file_helpers.open_image(image_path=mask_path, mode="MASK")

        # Re-do the augmentation until the mask is valid.
        mask_is_valid = False
        while not mask_is_valid:
            transformed = self.transform({"image": image, "mask": mask})
            mask_is_valid = self.is_mask_valid(transformed["mask"].numpy())

        # Get binary masks.
        target = self.get_binary_masks(transformed["mask"])

        return {
            "image_path": str(image_path),  # Str for torch dataloader compatibility.
            "image": transformed["image"],
            "mask": transformed["mask"],
            "target": target,
        }


class MaskSemanticSegmentationDatasetArgs(PydanticConfig):
    image_dir: Path
    mask_dir: Path
    classes: dict[int, str] | None = None
    ignore_classes: set[int] = set()
    check_empty_targets: bool = True

    # NOTE(Guarin, 07/25): The interface with below methods is experimental. Not yet
    # sure if it makes sense to have this in dataset args.
    def list_image_filenames(self) -> Iterable[ImageFilename]:
        for image_filename in file_helpers.list_image_filenames(
            image_dir=self.image_dir
        ):
            if (self.mask_dir / image_filename).with_suffix(".png").exists():
                yield image_filename

    @staticmethod
    def get_dataset_cls() -> type[MaskSemanticSegmentationDataset]:
        return MaskSemanticSegmentationDataset


class SplitArgs(PydanticConfig):
    images: PathLike
    masks: PathLike


class MaskSemanticSegmentationDataArgs(TaskDataArgs):
    train: SplitArgs
    val: SplitArgs
    classes: dict[int, str]
    ignore_classes: set[int] = set()
    check_empty_targets: bool = True

    # NOTE(Guarin, 07/25): The interface with below methods is experimental. Not yet
    # sure if this makes sense to have in data args.
    def get_train_args(self) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.train.images),
            mask_dir=Path(self.train.masks),
            classes=self.classes,
            ignore_classes=self.ignore_classes,
            check_empty_targets=self.check_empty_targets,
        )

    def get_val_args(self) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.val.images),
            mask_dir=Path(self.val.masks),
            classes=self.classes,
            ignore_classes=self.ignore_classes,
            check_empty_targets=self.check_empty_targets,
        )
