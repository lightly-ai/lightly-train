#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import albumentations as A
import pytest
import torch
from torch import Tensor

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)

from .. import helpers


class DummyTransform(TaskTransform):
    def __init__(self, transform_args: TaskTransformArgs):
        super().__init__(transform_args=transform_args)
        self.transform = A.Compose(
            [
                A.Resize(32, 32),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.pytorch.transforms.ToTensorV2(),
            ]
        )

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        output: TaskTransformOutput = self.transform(**input)
        return output


class TestMaskSemanticSegmentationDataset:
    @pytest.mark.parametrize(
        "num_classes, expected_mask_dtype",
        [
            (5, torch.uint8),
            (500, torch.int32),
        ],
    )
    def test__getitem__(
        self, num_classes: int, expected_mask_dtype: torch.dtype, tmp_path: Path
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg", "image1.jpg"]
        mask_filenames = ["image0.png", "image1.png"]
        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=num_classes)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            classes={0: "background", 1: "object"},
        )
        transform = DummyTransform(transform_args=TaskTransformArgs())
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_filenames=list(dataset_args.list_image_filenames()),
            transform=transform,
        )

        assert len(dataset) == 2
        for item in dataset:  # type: ignore[attr-defined]
            assert isinstance(item["image"], Tensor)
            assert item["image"].shape == (3, 32, 32)
            assert item["image"].dtype == torch.float32
            assert isinstance(item["mask"], Tensor)
            assert item["mask"].shape == (32, 32)
            assert item["mask"].dtype == expected_mask_dtype
            # Need conversion to int because min/max is not implemented for uint16.
            assert item["mask"].min() >= 0
            assert item["mask"].max() < num_classes
        assert sorted(item["image_path"] for item in dataset) == [  # type: ignore[attr-defined]
            str(image_dir / "image0.jpg"),
            str(image_dir / "image1.jpg"),
        ]

    def test_mask_unique_values_from_train_task_setup(self, tmp_path: Path) -> None:
        train_images = tmp_path / "train_images"
        train_masks = tmp_path / "train_masks"
        val_images = tmp_path / "val_images"
        val_masks = tmp_path / "val_masks"

        helpers.create_images(train_images)
        helpers.create_masks(train_masks)
        helpers.create_images(val_images)
        helpers.create_masks(val_masks)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=train_images,
            mask_dir=train_masks,
            classes={0: "background", 1: "car"},
        )

        transform = DummyTransform(transform_args=TaskTransformArgs())
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_filenames=list(dataset_args.list_image_filenames()),
            transform=transform,
        )

        for idx in range(len(dataset)):
            item = dataset[idx]
            mask = item["mask"]

            assert isinstance(mask, torch.Tensor), "Mask is not a tensor"
            unique_vals = mask.unique().tolist()
            print(f"Mask {idx} unique values: {unique_vals}")

            # Check values are within class range
            assert all(v in {0, 1} for v in unique_vals), (
                f"Unexpected mask values: {unique_vals}"
            )
