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
from typing import ClassVar, Dict, Literal, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field, TypeAdapter, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._env import Env
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import (
    BinaryMasksDict,
    ImageFilename,
    MaskSemanticSegmentationDatasetItem,
    PathLike,
)


class ClassInfo(PydanticConfig):
    name: str
    values: set[int] = Field(strict=False)


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
        self.ignore_index = dataset_args.ignore_index

        # Get the class mapping.
        self.class_mapping = self.get_class_mapping()
        self.valid_classes = np.array(list(self.class_mapping.keys()))

        image_mode = Env.LIGHTLY_TRAIN_IMAGE_MODE.value
        if image_mode not in ("RGB", "UNCHANGED"):
            raise ValueError(
                f'Invalid image mode: {Env.LIGHTLY_TRAIN_IMAGE_MODE.name}="{image_mode}". '
                "Supported modes are 'RGB' and 'UNCHANGED'."
            )
        self.image_mode: Literal["RGB", "UNCHANGED"] = image_mode  # type: ignore[assignment]

        # Optionally filter image filenames corresponding to empty targets.
        if dataset_args.check_empty_targets:
            self.filter_empty_targets()

    def is_mask_valid(self, mask: NDArray[np.uint8]) -> bool:
        # Get unique values in the mask.
        unique_values = np.unique(mask)

        # Check if at least one value in the mask is in the valid classes.
        return bool(np.isin(unique_values, self.valid_classes).any())

    def filter_empty_targets(self) -> None:
        # TODO(Thomas, 07/25): Move the filtering outside of the dataset for compatibility
        # with mmapped files and speed.
        # Instantiate new list of file names
        new_image_filenames = []

        # Populate the new lists with file names corresponding to valid targets.
        for filename in self.image_filenames:
            filepath = (self.args.mask_dir / filename).with_suffix(".png")

            mask = file_helpers.open_image_numpy(image_path=filepath, mode="MASK")
            if self.is_mask_valid(mask):
                new_image_filenames.append(filename)

        # Display the number of filtered files.
        # TODO(Thomas, 07/25): Change the print to logging once the function is moved
        # outside of the dataset.
        n_filtered_files = len(self) - len(new_image_filenames)
        print(f"Filtered {n_filtered_files} invalid masks out of {len(self)}.")

        # Update the list of valid files.
        self.image_filenames = new_image_filenames

    def get_class_mapping(self) -> dict[int, int]:
        ignore_classes = self.args.ignore_classes or set()
        return {
            class_id: i
            for i, class_id in enumerate(
                class_id
                for class_id in self.args.classes.keys()
                if class_id not in ignore_classes
            )
        }

    def __len__(self) -> int:
        return len(self.image_filenames)

    def get_binary_masks(self, mask: Tensor) -> BinaryMasksDict:
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
            img_labels.append(self.class_mapping[class_id])

        binary_masks: BinaryMasksDict = {
            "masks": torch.stack(img_masks),
            "labels": mask.new_tensor(img_labels, dtype=torch.long),
        }
        return binary_masks

    def remap_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # Create a lookup table initialized with ignore_index
        max_class = int(mask.max().item())
        lut = mask.new_full((max_class + 1,), self.ignore_index, dtype=torch.long)

        # Fill in valid mappings
        for old_class, new_class in self.class_mapping.items():
            if old_class <= max_class:
                lut[old_class] = new_class

        # Use LUT to remap efficiently
        return lut[mask.to(torch.long)]

    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        image_filename = self.image_filenames[index]
        image_path = self.args.image_dir / image_filename
        mask_path = (self.args.mask_dir / image_filename).with_suffix(".png")

        # Load the image and the mask.
        image = file_helpers.open_image_numpy(
            image_path=image_path, mode=self.image_mode
        )
        mask = file_helpers.open_image_numpy(image_path=mask_path, mode="MASK")

        # Verify that the mask and the image have the same shape.
        assert image.shape[:2] == mask.shape, (
            f"Shape mismatch: image shape is {image.shape[:2]} while mask shape is {mask.shape}."
        )

        # Re-do the augmentation until the mask is valid.
        mask_is_valid = False
        for _ in range(20):
            # (H, W, C) -> (C, H, W)
            transformed = self.transform({"image": image, "mask": mask})
            mask_is_valid = self.is_mask_valid(transformed["mask"].numpy())
            if mask_is_valid:
                break

        # Raise an error if the mask is still empty.
        if not mask_is_valid:
            raise RuntimeError(
                "Failed to obtain a valid mask after 20 augmentation retries. "
                "Consider enabling `check_empty_targets=True` in the data arguments to "
                "filter out such samples before training."
            )

        # Get binary masks.
        # TODO(Thomas, 07/25): Make this optional.
        binary_masks = self.get_binary_masks(transformed["mask"])

        # Mark pixels to ignore in the masks.
        # TODO(Thomas, 07/25): Make this optional.
        transformed_mask = transformed["mask"]
        transformed_mask = self.remap_mask(transformed_mask)

        return {
            "image_path": str(image_path),  # Str for torch dataloader compatibility.
            "image": transformed["image"],
            "mask": transformed_mask,
            "binary_masks": binary_masks,
        }


class MaskSemanticSegmentationDatasetArgs(PydanticConfig):
    image_dir: Path
    mask_dir: Path
    classes: dict[int, ClassInfo]
    # Disable strict to allow pydantic to convert lists/tuples to sets.
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    check_empty_targets: bool = True
    ignore_index: int

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
    ignore_index: ClassVar[int] = -100
    train: SplitArgs
    val: SplitArgs
    classes: dict[int, ClassInfo]
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    check_empty_targets: bool = True

    @field_validator("classes", mode="before")
    @classmethod
    def validate_classes(
        cls, classes: dict[int, str | dict[str, str | Sequence[int]]]
    ) -> dict[int, ClassInfo]:
        # Let Pydantic validate the structure and types
        classes_validated = TypeAdapter(
            Dict[int, Union[str, ClassInfo]]
        ).validate_python(classes)

        # Convert to ClassInfo objects
        class_infos: dict[int, ClassInfo] = {}
        for class_id, class_info in classes_validated.items():
            if isinstance(class_info, str):
                class_infos[class_id] = ClassInfo(name=class_info, values={class_id})
            else:
                class_infos[class_id] = class_info

        # Check the class mappings for validity.
        class_values: set[int] = set()

        for class_id, class_info in class_infos.items():
            for value in class_info.values:
                # Check for multiple values across different class mappings
                if value in class_values:
                    raise ValueError(
                        f"Invalid class mapping: Class {value} appears in multiple class definitions. "
                        f"Each old class value can only mapped to one new class.\n\n"
                        f"INCORRECT (class {value} is duplicated):\n"
                        f"classes = {{\n"
                        f"  255: {{'name': 'background-255', 'values': [0, 1, 2]}},\n"
                        f"  254: {{'name': 'background-254', 'values': [0, 1, 2]}}  # <- class [0, 1, 2] conflict with class 255\n"
                        f"}}\n\n"
                        f"CORRECT (each set of class values belongs to only one class):\n"
                        f"classes = {{\n"
                        f"  255: {{'name': 'background-255', 'values': [0, 1, 2]}},\n"
                        f"  254: {{'name': 'background-254', 'values': [3, 4, 5]}}  # <- unique values\n"
                        f"}}"
                    )
                class_values.add(value)

        return class_infos

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns classes (AFTER mapping) that are not ignored with the name."""
        ignore_classes = set() if self.ignore_classes is None else self.ignore_classes

        result = {}
        for class_id, class_info in self.classes.items():
            if class_id not in ignore_classes:
                result[class_id] = class_info.name

        return result

    @property
    def num_included_classes(self) -> int:
        return len(self.included_classes)

    # NOTE(Guarin, 07/25): The interface with below methods is experimental. Not yet
    # sure if this makes sense to have in data args.
    def get_train_args(
        self,
    ) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.train.images),
            mask_dir=Path(self.train.masks),
            classes=self.classes,
            ignore_classes=self.ignore_classes,
            check_empty_targets=self.check_empty_targets,
            ignore_index=self.ignore_index,
        )

    def get_val_args(
        self,
    ) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.val.images),
            mask_dir=Path(self.val.masks),
            classes=self.classes,
            ignore_classes=self.ignore_classes,
            check_empty_targets=self.check_empty_targets,
            ignore_index=self.ignore_index,
        )
