#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import torch

from lightly_train._data.yolo_instance_segmentation_dataset import (
    YOLOInstanceSegmentationDataArgs,
    YOLOInstanceSegmentationDataset,
)
from lightly_train._transforms.instance_segmentation_transform import (
    InstanceSegmentationTransform,
    InstanceSegmentationTransformArgs,
)

from .. import helpers


class TestYOLOInstanceSegmentationDataset:
    def test__split_first(self, tmp_path: Path) -> None:
        helpers.create_yolo_instance_segmentation_dataset(
            tmp_path=tmp_path, split_first=True, num_files=2, height=64, width=128
        )

        args = YOLOInstanceSegmentationDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={1: "class_0", 2: "class_1"},
        )
        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOInstanceSegmentationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=list(train_args.list_image_info()),
        )

        val_dataset = YOLOInstanceSegmentationDataset(
            dataset_args=args.get_val_args(),
            transform=_get_transform(),
            image_info=list(val_args.list_image_info()),
        )

        assert len(train_dataset) == 2
        assert len(val_dataset) == 2

        sample = train_dataset[0]
        # Switch to float32 after normalization is added
        assert sample["image"].dtype == torch.uint8
        assert sample["image"].shape == (3, 64, 128)
        assert sample["binary_masks"]["masks"].dtype == torch.bool
        assert sample["binary_masks"]["masks"].shape == (1, 64, 128)
        assert sample["binary_masks"]["labels"].dtype == torch.long
        assert sample["binary_masks"]["labels"].shape == (1,)
        assert sample["bboxes"].dtype == torch.float
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].dtype == torch.long
        assert sample["classes"].shape == (1,)
        # Classes are mapped to internal class ids in [0, num_included_classes - 1]
        assert torch.all(sample["classes"] <= 1)

        sample = val_dataset[0]
        assert sample["image"].shape == (3, 64, 128)
        assert sample["binary_masks"]["masks"].shape == (1, 64, 128)
        assert sample["binary_masks"]["labels"].shape == (1,)
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].shape == (1,)

    def test__split_last(self, tmp_path: Path) -> None:
        helpers.create_yolo_instance_segmentation_dataset(
            tmp_path=tmp_path, split_first=False, num_files=2, height=64, width=128
        )

        args = YOLOInstanceSegmentationDataArgs(
            path=tmp_path,
            train="images/train",
            val="images/val",
            names={1: "class_0", 2: "class_1"},
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOInstanceSegmentationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=list(train_args.list_image_info()),
        )

        val_dataset = YOLOInstanceSegmentationDataset(
            dataset_args=args.get_val_args(),
            transform=_get_transform(),
            image_info=list(val_args.list_image_info()),
        )

        sample = train_dataset[0]
        # Switch to float32 after normalization is added
        assert sample["image"].dtype == torch.uint8
        assert sample["image"].shape == (3, 64, 128)
        assert sample["binary_masks"]["masks"].dtype == torch.bool
        assert sample["binary_masks"]["masks"].shape == (1, 64, 128)
        assert sample["binary_masks"]["labels"].dtype == torch.long
        assert sample["binary_masks"]["labels"].shape == (1,)
        assert sample["bboxes"].dtype == torch.float
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].dtype == torch.long
        assert sample["classes"].shape == (1,)
        # Classes are mapped to internal class ids in [0, num_included_classes - 1]
        assert torch.all(sample["classes"] <= 1)

        sample = val_dataset[0]
        assert sample["image"].shape == (3, 64, 128)
        assert sample["binary_masks"]["masks"].shape == (1, 64, 128)
        assert sample["binary_masks"]["labels"].shape == (1,)
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].shape == (1,)


def _get_transform() -> InstanceSegmentationTransform:
    transform_args = InstanceSegmentationTransformArgs(
        num_channels="auto",
    )
    return InstanceSegmentationTransform(transform_args)
