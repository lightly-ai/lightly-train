#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Union

import numpy as np
import torch
from pydantic import Field, TypeAdapter, field_validator
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypeGuard

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._env import Env
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import (
    BinaryMasksDict,
    MaskSemanticSegmentationDatasetItem,
    NDArrayImage,
    PathLike,
)


class LabelsClassInfo(PydanticConfig):
    name: str
    labels: set[int] = Field(alias="values")

    @field_validator("labels", mode="before")
    @classmethod
    def normalize_labels(cls, v: set[int] | list[int] | int) -> set[int]:
        # Case for a single label or a set of labels: 0 -> {0} or {0, 1, 2} -> {0, 1, 2}
        if isinstance(v, set):
            return v
        # List of labels: [0, 1, 2] -> {0, 1, 2}
        elif isinstance(v, list):
            return set(v)
        # Single label: 0 -> {0}
        elif isinstance(v, int):
            return {v}
        else:
            raise ValueError(f"Expected int or list of ints, got {type(v)}")


class ColorsClassInfo(PydanticConfig):
    name: str
    colors: set[tuple[int, ...]] = Field(alias="values")

    @field_validator("colors", mode="before")
    @classmethod
    def normalize_colors(
        cls, v: tuple[int, ...] | list[tuple[int, ...]] | set[tuple[int, ...]]
    ) -> set[tuple[int, ...]]:
        # Case for a single color tuple: (0, 0, 0) -> {(0, 0, 0)}
        if isinstance(v, tuple):
            return {v}
        # List of color tuples: [(0, 0, 0), (255, 255, 255)] -> {(0, 0, 0), (255, 255, 255)}
        elif isinstance(v, list):
            return set(v)
        # Set of color tuples: {(0, 0, 0), (255, 255, 255)} -> {(0, 0, 0), (255, 255, 255)}
        elif isinstance(v, set):
            return v
        else:
            raise ValueError(f"Expected tuple or list of tuples, got {type(v)}")

    @field_validator("colors", mode="after")
    @classmethod
    def validate_rgb_colors(cls, colors: set[tuple[int, ...]]) -> set[tuple[int, ...]]:
        # Validate that each color is a valid RGB tuple with channels in [0, 255].
        for color in colors:
            if len(color) != 3:
                raise ValueError(
                    f"Invalid RGB color values: {color}. Values must be integers between 0 and 255."
                )
            r, g, b = color
            for channel in (r, g, b):
                if not isinstance(channel, int) or not (0 <= channel <= 255):
                    raise ValueError(
                        f"Invalid RGB color values: {color}. Values must be integers between 0 and 255."
                    )
        return colors


ClassInfo = Union[ColorsClassInfo, LabelsClassInfo]


def _are_colors_classes(
    classes: dict[int, ClassInfo],
) -> TypeGuard[dict[int, ColorsClassInfo]]:
    """TypeGuard ensuring all class infos are ColorsClassInfo.

    This allows mypy to narrow `classes` to `dict[int, ColorsClassInfo]` within
    the True-branch.
    """
    return all(
        isinstance(class_info, ColorsClassInfo) for class_info in classes.values()
    )


class MaskSemanticSegmentationDataset(Dataset[MaskSemanticSegmentationDatasetItem]):
    def __init__(
        self,
        dataset_args: MaskSemanticSegmentationDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: TaskTransform,
    ):
        self.args = dataset_args
        self.filepaths = image_info
        self.transform = transform
        self.ignore_index = dataset_args.ignore_index

        # Get the class mapping.
        self.class_mapping = self.get_class_mapping()
        self.valid_classes = torch.tensor(list(self.class_mapping.keys()))

        image_mode = Env.LIGHTLY_TRAIN_IMAGE_MODE.value
        if image_mode not in ("RGB", "UNCHANGED"):
            raise ValueError(
                f'Invalid image mode: {Env.LIGHTLY_TRAIN_IMAGE_MODE.name}="{image_mode}". '
                "Supported modes are 'RGB' and 'UNCHANGED'."
            )
        # Convert string to enum value
        if image_mode == "RGB":
            self.image_mode = ImageMode.RGB
        elif image_mode == "UNCHANGED":
            self.image_mode = ImageMode.UNCHANGED
        else:
            # This should not happen due to the check above, but added for type safety
            raise ValueError(f"Unexpected image mode: {image_mode}")

    def is_mask_valid(self, mask: Tensor) -> bool:
        # Check if at least one value in the mask is in the valid classes.
        unique_classes: Tensor = mask.unique()  # type: ignore[no-untyped-call]
        return bool(torch.isin(unique_classes, self.valid_classes).any())

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
        return len(self.filepaths)

    def get_binary_masks(self, mask: Tensor) -> BinaryMasksDict:
        # This follows logic from:
        # https://github.com/tue-mps/eomt/blob/716cbd562366b9746804579b48b866da487d9485/datasets/ade20k_semantic.py#L47-L48

        img_masks = []
        img_labels = []
        class_ids = mask.unique().tolist()  # type: ignore[no-untyped-call]

        # Iterate over the labels present in the mask.
        for class_id in class_ids:
            # Check if the class id is the valid classes.
            if class_id not in self.class_mapping:
                continue

            # Create binary mask for the class.
            img_masks.append(mask == class_id)

            # Store the class label.
            img_labels.append(self.class_mapping[class_id])

        binary_masks: BinaryMasksDict = {
            "masks": (
                torch.stack(img_masks)
                if img_masks
                else mask.new_zeros(size=(0, *mask.shape), dtype=torch.bool)
            ),
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

    def _map_rgb_masks_to_integer_masks(
        self, rgb_mask: NDArrayImage, class_infos: dict[int, ColorsClassInfo]
    ) -> NDArrayImage:
        """Map RGB mask to single channel mask using class labels from ColorsClassInfo."""
        # Initialize single channel mask with ignore_index

        invalid_index = self.valid_classes.max().item() + 1
        single_channel_mask = np.full(rgb_mask.shape[:2], invalid_index, dtype=np.uint8)

        # Map each RGB color to its corresponding class label
        for class_id, class_info in class_infos.items():
            for color in class_info.colors:
                # Find pixels that match this color
                mask = np.all(rgb_mask == color, axis=2)
                # Assign class_id to matching pixels
                single_channel_mask[mask] = class_id

        return single_channel_mask

    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        row = self.filepaths[index]

        image_path = row["image_filepaths"]
        mask_path = row["mask_filepaths"]

        # Load the image and the mask.
        image = file_helpers.open_image_numpy(
            image_path=Path(image_path), mode=self.image_mode
        )
        mask = file_helpers.open_image_numpy(
            image_path=Path(mask_path), mode=ImageMode.MASK
        )

        # Verify that the mask and the image have the same shape.
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch: image shape is {image.shape[:2]} while mask shape is {mask.shape}."
            )

        # Local alias to enable type narrowing with TypeGuard
        classes = self.args.classes

        # Check that if the mask is RGB, then the class info must be ColorsClassInfo
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            if not _are_colors_classes(classes):
                raise ValueError(
                    "Expected colors specified in `classes` for RGB masks but got labels. "
                    "For RGB masks, you have to specify the colors that correspond to each class.\n\n"
                    "The RGB colors should be provided as a tuple (not a list!) to `values`:\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'values': (0, 0, 0)},\n"
                    "  1: {'name': 'road', 'values': (128, 128, 128)}\n"
                    "}\n\n"
                    "Mapping multiple colors to a single class also by providing a list of tuples:\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'values': [(0, 0, 0), (255, 255, 255)]},\n"
                    "  1: {'name': 'road', 'values': [(128, 128, 128), (64, 64, 64)]}\n"
                    "}"
                )

            # Map RGB mask to single channel mask using class labels
            mask = self._map_rgb_masks_to_integer_masks(mask, classes)

        # Try to find an augmentation that contains a valid mask. This increases the
        # probability for a good training signal. If no valid mask is found we still
        # return the last transformed mask and proceed with training.
        for _ in range(20):
            # (H, W, C) -> (C, H, W)
            transformed = self.transform({"image": image, "mask": mask})
            if self.is_mask_valid(transformed["mask"]):
                break

        # Get binary masks.
        # TODO(Thomas, 07/25): Make this optional.
        binary_masks = self.get_binary_masks(transformed["mask"])

        # Mark pixels to ignore in the masks.
        # TODO(Thomas, 07/25): Make this optional.
        transformed_mask = self.remap_mask(transformed["mask"])

        return {
            "image_path": str(image_path),  # Str for torch dataloader compatibility.
            "image": transformed["image"],
            "mask": transformed_mask,
            "binary_masks": binary_masks,
        }


class MaskSemanticSegmentationDatasetArgs(PydanticConfig):
    image_dir: Path
    mask_dir_or_file: str
    classes: dict[int, ClassInfo]
    # Disable strict to allow pydantic to convert lists/tuples to sets.
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    ignore_index: int

    def list_image_info(self) -> Iterable[dict[str, str]]:
        is_mask_dir = Path(self.mask_dir_or_file).is_dir()
        for image_filepath in file_helpers.list_image_files(
            imgs_and_dirs=[self.image_dir]
        ):
            if is_mask_dir:
                mask_filepath = self.mask_dir_or_file / image_filepath.relative_to(
                    self.image_dir
                ).with_suffix(".png")
            else:
                mask_filepath = Path(
                    self.mask_dir_or_file.format(
                        image_path=image_filepath,
                    )
                )

            if mask_filepath.exists():
                yield {
                    "image_filepaths": str(image_filepath),
                    "mask_filepaths": str(mask_filepath),
                }

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

    @field_validator("classes", mode="before")
    @classmethod
    def validate_classes(
        cls, classes: dict[int, str | dict[str, Any]]
    ) -> dict[int, ClassInfo]:
        classes_validated = TypeAdapter(
            Dict[int, Union[str, LabelsClassInfo, ColorsClassInfo]]
        ).validate_python(classes)

        # Convert to ClassInfo objects and perform consistency checks.
        class_infos: dict[int, ClassInfo] = {}
        class_types: set[type] = set()
        class_labels: set[int] = set()
        class_colors: set[tuple[int, ...]] = set()

        for class_id, class_info in classes_validated.items():
            if isinstance(class_info, str):
                class_info = LabelsClassInfo(name=class_info, values={class_id})

            # Check for inconsistent class types early
            class_types.add(type(class_info))
            if len(class_types) > 1:
                raise ValueError(
                    "All classes must be consistently either LabelsClassInfo or ColorsClassInfo. Mixing types is not allowed."
                )

            if isinstance(class_info, LabelsClassInfo):
                for label in class_info.labels:
                    # Check for multiple labels across different class mappings
                    if label in class_labels:
                        raise ValueError(
                            f"Invalid class mapping: Class label {label} appears in multiple class definitions. "
                            f"Each old class label can only mapped to one new class.\n\n"
                            f"INCORRECT (class label {label} is duplicated):\n"
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
                    class_labels.add(label)
            elif isinstance(class_info, ColorsClassInfo):
                for color in class_info.colors:
                    # Check for multiple colors across different class mappings
                    if color in class_colors:
                        raise ValueError(
                            f"Invalid class mapping: Class color {color} appears in multiple class definitions. "
                            f"Each RGB color in the mask can only be mapped to one new class.\n\n"
                            f"INCORRECT (RGB color {color} is duplicated):\n"
                            f"classes = {{\n"
                            f"  0: {{'name': 'background', 'values': [(0, 0, 0), (255, 255, 255)]}},\n"
                            f"  1: {{'name': 'road', 'values': [(0, 0, 0), (128, 128, 128)]}}  # <- color (0, 0, 0) conflict with class 0\n"
                            f"}}\n\n"
                            f"CORRECT (each RGB color belongs to only one class):\n"
                            f"classes = {{\n"
                            f"  0: {{'name': 'background', 'values': [(0, 0, 0), (255, 255, 255)]}},\n"
                            f"  1: {{'name': 'road', 'values': [(128, 128, 128), (64, 64, 64)]}}  # <- unique colors\n"
                            f"}}"
                        )
                    class_colors.add(color)
            else:
                pass

            class_infos[class_id] = class_info

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
            mask_dir_or_file=str(self.train.masks),
            classes=self.classes,
            ignore_classes=self.ignore_classes,
            ignore_index=self.ignore_index,
        )

    def get_val_args(
        self,
    ) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.val.images),
            mask_dir_or_file=str(self.val.masks),
            classes=self.classes,
            ignore_classes=self.ignore_classes,
            ignore_index=self.ignore_index,
        )
