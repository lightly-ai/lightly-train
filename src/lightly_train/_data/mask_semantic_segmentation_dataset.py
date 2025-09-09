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
from pydantic import AliasChoices, Field, TypeAdapter, field_validator
from torch import Tensor
from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._env import Env
from lightly_train._transforms.semantic_segmentation_transform import (
    SemanticSegmentationTransform,
    SemanticSegmentationTransformArgs,
)
from lightly_train.types import (
    BinaryMasksDict,
    MaskSemanticSegmentationDatasetItem,
    NDArrayImage,
    PathLike,
)


class SingleChannelClassInfo(PydanticConfig):
    name: str
    labels: set[int] = Field(
        validation_alias=AliasChoices("labels", "values"),
        serialization_alias="labels",
        strict=False,
    )


class MultiChannelClassInfo(PydanticConfig):
    name: str
    labels: set[tuple[int, ...]] = Field(
        validation_alias=AliasChoices("labels", "values"),
        serialization_alias="labels",
        strict=False,
    )


ClassInfo = Union[MultiChannelClassInfo, SingleChannelClassInfo]


class MaskSemanticSegmentationDataset(Dataset[MaskSemanticSegmentationDatasetItem]):
    def __init__(
        self,
        dataset_args: MaskSemanticSegmentationDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: SemanticSegmentationTransform,
    ):
        self.args = dataset_args
        self.filepaths = image_info
        self.transform = transform
        self.ignore_index = dataset_args.ignore_index

        # Get the class mapping.
        self.class_mapping = self.get_class_mapping()
        self.valid_classes = torch.tensor(list(self.class_mapping.keys()))

        transform_args = transform.transform_args
        assert isinstance(transform_args, SemanticSegmentationTransformArgs)

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

    def map_class_id_to_internal_class_id(self, mask: torch.Tensor) -> torch.Tensor:
        # Map only non-ignored pixels through the LUT and keep ignored pixels untouched.
        # NOTE: this relies on the fact that all `old_class`es are non-negative.
        ignore = mask == self.ignore_index
        valid = ~ignore

        if ignore.all():
            return mask  # entire mask is ignore; nothing to remap

        max_class = int(mask[valid].max().item())
        lut = mask.new_full((max_class + 1,), self.ignore_index, dtype=torch.long)

        # Fill in valid mappings
        for old_class, new_class in self.class_mapping.items():
            if old_class <= max_class:
                lut[old_class] = new_class

        mask = mask.to(torch.long)
        mask[valid] = lut[mask[valid]]

        return mask

    def map_mask_labels_to_class_ids(
        self,
        mask_with_labels: NDArrayImage,
    ) -> NDArrayImage:
        class_infos = self.args.classes
        # Always compare against a 3D mask: expand (H, W) -> (H, W, 1)
        mask_with_labels = (
            mask_with_labels
            if mask_with_labels.ndim == 3
            else mask_with_labels[:, :, np.newaxis]
        )
        # Initialize output single-channel mask with ignore_index
        mask_with_class_ids = np.full(
            mask_with_labels.shape[:2], self.ignore_index, dtype=np.int_
        )

        # Map labels (ints or tuples) to class ids
        for class_id, class_info in class_infos.items():
            for label in class_info.labels:
                # Normalize integer labels to 1-tuple for broadcasting with (H, W, 1)
                label_tuple = (label,) if isinstance(label, np.int_) else label
                # Find pixels that match this value across channels
                label_mask = np.all(mask_with_labels == label_tuple, axis=2)
                # Assign class_id to matching pixels
                mask_with_class_ids[label_mask] = class_id

        return mask_with_class_ids

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
                f"Shape mismatch: image (height, width) is {image.shape[:2]} while mask (height, width) is {mask.shape[:2]}."
            )

        # Local alias to enable type narrowing with TypeGuard
        classes = self.args.classes

        # Check that if the mask is multi-channel, then the class info must be MultiChannelClassInfo
        if len(mask.shape) == 3:
            if not all(
                isinstance(class_info, MultiChannelClassInfo)
                for class_info in classes.values()
            ):
                raise ValueError(
                    "Expected tuple labels specified in `classes` for multi-channel masks but got single-channel integer labels. "
                    "For multi-channel masks, you have to specify the tuple label value of the pixels that correspond to each class id.\n\n"
                    "The tuple labels must be provided as a list of tuples to `labels`:\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'labels': [(0, 0, 0)]},\n"
                    "  1: {'name': 'road', 'labels': [(0, 128, 128)]}\n"
                    "}\n\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'labels': [(0, 0, 0), (255, 255, 255)]},\n"
                    "  1: {'name': 'road', 'labels': [(0, 128, 128), (64, 64, 64)]}\n"
                    "}\n\n"
                    "Note: the key `values` is still accepted as an alias for `labels` for backward compatibility."
                )

        # Map mask labels (single- or multi-channel) to single channel class ids
        mask = self.map_mask_labels_to_class_ids(mask)

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
        transformed_mask = self.map_class_id_to_internal_class_id(transformed["mask"])

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
        mask_dir = Path(self.mask_dir_or_file)
        is_mask_dir = mask_dir.is_dir()
        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / image_filename
            if is_mask_dir:
                mask_filepath = (mask_dir / image_filename).with_suffix(".png")
            else:
                mask_filepath = Path(
                    self.mask_dir_or_file.format(image_path=image_filepath)
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
            Dict[int, Union[str, SingleChannelClassInfo, MultiChannelClassInfo]]
        ).validate_python(classes)

        # Convert to ClassInfo objects and perform consistency checks.
        class_infos: dict[int, ClassInfo] = {}
        class_types: set[type] = set()
        class_labels: set[int | tuple[int, ...]] = set()

        for class_id, class_info in classes_validated.items():
            if isinstance(class_info, str):
                class_info = SingleChannelClassInfo(name=class_info, labels={class_id})

            # Check for inconsistent class types early
            class_types.add(type(class_info))
            if len(class_types) > 1:
                raise ValueError(
                    "Invalid class mapping: mixed class types detected. "
                    "All labels must be consistently either integers for single-channel masks or tuples for multi-channel masks. Mixed types are not allowed.\n\n"
                    "INCORRECT (mixed integers and tuples):\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'labels': [0, 1, 2]},\n"
                    "  1: {'name': 'road', 'labels': [(0, 0, 0), (128, 128, 128)]}  # <- mixed types\n"
                    "}\n\n"
                    "CORRECT (use only integers):\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'labels': [0, 1, 2]},\n"
                    "  1: {'name': 'road', 'labels': [3, 4, 5]}\n"
                    "}\n\n"
                    "CORRECT (use only tuples):\n"
                    "classes = {\n"
                    "  0: {'name': 'background', 'labels': [(0, 0, 0), (255, 255, 255)]},\n"
                    "  1: {'name': 'road', 'labels': [(128, 128, 128), (64, 64, 64)]}\n"
                    "}\n\n"
                    "Note: the key `values` is still accepted as an alias for `labels` for backward compatibility."
                )

            for label in class_info.labels:
                # Check for multiple labels across different class mappings
                if label in class_labels:
                    if isinstance(class_info, SingleChannelClassInfo):
                        raise ValueError(
                            f"Invalid class mapping: integer label {label} appears in multiple class definitions. "
                            "Each integer label in the single-channel masks can only be mapped to one class id.\n\n"
                            f"INCORRECT (integer label {label} is duplicated):\n"
                            "classes = {{\n"
                            "  0: {{'name': 'background', 'labels': [0, 1, 2]}},\n"
                            "  1: {{'name': 'road', 'labels': [0, 3, 4]}}  # <- integer label 0 conflict with class id 0\n"
                            "}}\n\n"
                            "CORRECT (each integer label belongs to only one class id):\n"
                            "classes = {{\n"
                            "  0: {{'name': 'background', 'labels': [0, 1, 2]}},\n"
                            "  1: {{'name': 'road', 'labels': [3, 4, 5]}}  # <- unique integer labels\n"
                            "}}\n\n"
                            "Note: you can also use the key `values` as an alias for `labels`."
                        )
                    else:
                        raise ValueError(
                            f"Invalid class mapping: tuple label {label} appears in multiple class definitions. "
                            "Each tuple label in the multi-channel masks can only be mapped to one class id.\n\n"
                            f"INCORRECT (tuple label {label} is duplicated):\n"
                            "classes = {{\n"
                            "  0: {{'name': 'background', 'labels': [(0, 0, 0), (255, 255, 255)]}},\n"
                            "  1: {{'name': 'road', 'labels': [(0, 0, 0), (128, 128, 128)]}}  # <- tuple label (0, 0, 0) conflict with class id 0\n"
                            "}}\n\n"
                            "CORRECT (each tuple label belongs to only one class id):\n"
                            "classes = {{\n"
                            "  0: {{'name': 'background', 'labels': [(0, 0, 0), (255, 255, 255)]}},\n"
                            "  1: {{'name': 'road', 'labels': [(128, 128, 128), (64, 64, 64)]}}  # <- unique tuple labels\n"
                            "}}\n\n"
                            "Note: the key `values` is still accepted as an alias for `labels`."
                        )
                class_labels.add(label)

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
