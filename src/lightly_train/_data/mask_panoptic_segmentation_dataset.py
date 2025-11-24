#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Iterable

import numpy as np
import torch
from pydantic import Field
from torch import Tensor

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._data.task_batch_collation import (
    BaseCollateFunction,
    MaskPanopticSegmentationCollateFunction,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._env import Env
from lightly_train._transforms.panoptic_segmentation_transform import (
    PanopticSegmentationTransform,
    PanopticSegmentationTransformArgs,
)
from lightly_train.types import (
    MaskPanopticSegmentationDatasetItem,
    PanopticBinaryMasksDict,
    PathLike,
)


class SingleChannelClassInfo(PydanticConfig):
    name: str
    labels: set[int] = Field(strict=False)
    is_thing: bool


ClassInfo = SingleChannelClassInfo


class MaskPanopticSegmentationDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: MaskPanopticSegmentationDatasetArgs

    batch_collate_fn_cls: ClassVar[type[BaseCollateFunction]] = (
        MaskPanopticSegmentationCollateFunction
    )

    def __init__(
        self,
        dataset_args: MaskPanopticSegmentationDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: PanopticSegmentationTransform,
    ):
        super().__init__(
            transform=transform, dataset_args=dataset_args, image_info=image_info
        )

        # Get the class mapping.
        ignore_classes = dataset_args.ignore_classes or set()
        class_id_to_internal_class_id: dict[int, int] = {}
        internal_class_id = 0
        for class_id in dataset_args.stuff_classes.keys():
            if class_id not in ignore_classes:
                class_id_to_internal_class_id[class_id] = internal_class_id
                internal_class_id += 1
        for class_id in dataset_args.thing_classes.keys():
            if class_id not in ignore_classes:
                class_id_to_internal_class_id[class_id] = internal_class_id
                internal_class_id += 1

        # Internal class ids are structured as follows:
        # [0, num_stuff_classes - 1] -> stuff classes
        # [num_stuff_classes, num_stuff_classes + num_thing_classes - 1] -> thing classes
        # NOTE: This must match the implementations in the train and task models!
        self.class_id_to_internal_class_id = class_id_to_internal_class_id
        # Special class id for pixels that are not assigned to any class in the dataset.
        # NOTE: This must match the implementation in the task model!
        self.internal_ignore_class_id = len(class_id_to_internal_class_id)

        transform_args = transform.transform_args
        assert isinstance(transform_args, PanopticSegmentationTransformArgs)

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

    def __getitem__(self, index: int) -> MaskPanopticSegmentationDatasetItem:
        row = self.image_info[index]

        image_path = row["image_filepaths"]
        mask_path = row["mask_filepaths"]
        segments = json.loads(row["segments"])
        segment_id_to_segment = {segment["id"]: segment for segment in segments}

        # Load the image and the mask.
        image = file_helpers.open_image_numpy(
            image_path=Path(image_path), mode=self.image_mode
        )
        mask = file_helpers.open_mask_numpy(mask_path=Path(mask_path))

        # Verify that the mask and the image have the same shape.
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch: image (height, width) is {image.shape[:2]} while mask (height, width) is {mask.shape[:2]}."
            )

        mask = mask.astype(np.int_)
        if mask.ndim == 3:
            # Convert RGB encoded mask to single channel.
            # From https://cocodataset.org/#format-data:
            # > Note that when you load the PNG as an RGB image, you will need to compute
            # > the ids via ids=R+G*256+B*256^2
            mask = mask[..., 0] + mask[..., 1] * 256 + mask[..., 2] * (256**2)
        if mask.ndim != 2:
            raise ValueError(
                f"Invalid mask with shape {mask.shape}. Expected 2D array."
            )

        # Try to find an augmentation that contains a valid mask. This increases the
        # probability for a good training signal. If no valid mask is found we still
        # return the last transformed mask and proceed with training.
        for _ in range(20):
            # (H, W, C) -> (C, H, W)
            transformed = self.transform({"image": image, "mask": mask})
            binary_masks = self.get_binary_masks(
                transformed["mask"], segment_id_to_segment=segment_id_to_segment
            )
            if self.is_valid_binary_masks(binary_masks):
                break

        masks = self.get_masks(binary_masks)

        return {
            "image_path": str(image_path),  # Str for torch dataloader compatibility.
            "image": transformed["image"],
            "masks": masks,
            "binary_masks": binary_masks,
        }

    def get_binary_masks(
        self, mask: Tensor, segment_id_to_segment: dict[int, dict[Any, Any]]
    ) -> PanopticBinaryMasksDict:
        masks = []
        labels = []
        iscrowd = []
        for segment_id in mask.unique().tolist():  # type: ignore
            segment = segment_id_to_segment.get(segment_id)
            if segment is None:
                # Unknown segment id, skip.
                continue
            class_id = segment["category_id"]
            internal_class_id = self.class_id_to_internal_class_id.get(class_id)
            if internal_class_id is None:
                # Ignored class.
                continue
            masks.append(mask == segment_id)
            labels.append(internal_class_id)
            iscrowd.append(
                segment.get("iscrowd", False) if segment is not None else False
            )

        binary_masks: PanopticBinaryMasksDict = {
            "masks": (
                torch.stack(masks)
                if masks
                else mask.new_zeros(size=(0, *mask.shape), dtype=torch.bool)
            ),
            "labels": mask.new_tensor(labels, dtype=torch.long),
            "iscrowd": mask.new_tensor(iscrowd, dtype=torch.bool),
        }
        return binary_masks

    def get_masks(self, binary_masks: PanopticBinaryMasksDict) -> Tensor:
        """Convert binary masks to panoptic segmentation masks.

        Returns:
            (H, W, 2) Tensor where the last dimension contains (label, segment_id).
            Segment ids are in [0, num_segments-1]. -1 indicates no segment.
        """
        binary_mask = binary_masks["masks"]
        N, H, W = binary_mask.shape
        # Initialize with:
        # - label = ignore class id
        # - segment id = -1
        masks = binary_mask.new_full((H, W, 2), fill_value=-1, dtype=torch.int)
        masks[..., 0] = self.internal_ignore_class_id
        for i in range(N):
            binary_mask = binary_masks["masks"][i]
            label = binary_masks["labels"][i]
            masks[binary_mask, 0] = label
            masks[binary_mask, 1] = i
        return masks

    def is_valid_binary_masks(self, binary_masks: PanopticBinaryMasksDict) -> bool:
        return len(binary_masks["labels"]) > 0


class MaskPanopticSegmentationDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    mask_dir_or_file: str
    annotation_file: Path
    thing_classes: dict[int, str]
    stuff_classes: dict[int, str]
    # Disable strict to allow pydantic to convert lists/tuples to sets.
    ignore_classes: set[int] | None = Field(default=None, strict=False)

    def list_image_info(self) -> Iterable[dict[str, str]]:
        mask_dir = Path(self.mask_dir_or_file)
        annotations = _load_annotations(self.annotation_file)
        # Mapping from file stem (filename without extension) to segments info.
        # We use the stem because images and masks can have different extensions and
        # users might either save the image filename or the mask filenames in the
        # annotations.
        file_stem_to_segments = {
            os.path.splitext(ann["file_name"])[0]: ann["segments_info"]
            for ann in annotations["annotations"]
        }
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
                    "segments": json.dumps(file_stem_to_segments[mask_filepath.stem]),
                }

    @staticmethod
    def get_dataset_cls() -> type[MaskPanopticSegmentationDataset]:
        return MaskPanopticSegmentationDataset


class SplitArgs(PydanticConfig):
    images: PathLike
    masks: PathLike
    annotations: PathLike


class MaskPanopticSegmentationDataArgs(TaskDataArgs):
    train: SplitArgs
    val: SplitArgs
    ignore_classes: set[int] | None = Field(default=None, strict=False)

    def train_imgs_path(self) -> Path:
        return Path(self.train.images)

    def val_imgs_path(self) -> Path:
        return Path(self.val.images)

    @cached_property
    def classes(self) -> dict[int, ClassInfo]:
        return _load_classes(annotation_file=Path(self.train.annotations))

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

    @property
    def thing_classes(self) -> dict[int, str]:
        return {
            class_id: class_name
            for class_id, class_name in self.included_classes.items()
            if self.classes[class_id].is_thing
        }

    @property
    def stuff_classes(self) -> dict[int, str]:
        return {
            class_id: class_name
            for class_id, class_name in self.included_classes.items()
            if not self.classes[class_id].is_thing
        }

    # NOTE(Guarin, 07/25): The interface with below methods is experimental. Not yet
    # sure if this makes sense to have in data args.

    def get_train_args(
        self,
    ) -> MaskPanopticSegmentationDatasetArgs:
        return MaskPanopticSegmentationDatasetArgs(
            image_dir=Path(self.train.images),
            mask_dir_or_file=str(self.train.masks),
            annotation_file=Path(self.train.annotations),
            thing_classes=self.thing_classes,
            stuff_classes=self.stuff_classes,
            ignore_classes=self.ignore_classes,
        )

    def get_val_args(
        self,
    ) -> MaskPanopticSegmentationDatasetArgs:
        return MaskPanopticSegmentationDatasetArgs(
            image_dir=Path(self.val.images),
            mask_dir_or_file=str(self.val.masks),
            annotation_file=Path(self.val.annotations),
            thing_classes=self.thing_classes,
            stuff_classes=self.stuff_classes,
            ignore_classes=self.ignore_classes,
        )


def _load_annotations(annotation_file: Path) -> dict[str, Any]:
    with annotation_file.open("r") as f:
        annotations = json.load(f)
    return annotations  # type: ignore


def _load_classes(annotation_file: Path) -> dict[int, ClassInfo]:
    annotations = _load_annotations(annotation_file)
    classes: dict[int, ClassInfo] = {}
    for class_info in annotations["categories"]:
        class_id = class_info["id"]
        classes[class_id] = ClassInfo(
            name=class_info["name"],
            labels={class_id},
            is_thing=bool(class_info["isthing"]),
        )
    return classes
