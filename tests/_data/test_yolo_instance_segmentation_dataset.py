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
from albumentations import BboxParams

from lightly_train._data.yolo_instance_segmentation_dataset import (
    YOLOInstanceSegmentationDataArgs,
    YOLOInstanceSegmentationDataset,
)
from lightly_train._transforms.instance_segmentation_transform import (
    InstanceSegmentationTransform,
    InstanceSegmentationTransformArgs,
)
from lightly_train._transforms.transform import NormalizeArgs

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
            names={0: "class_0", 1: "class_1"},
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
        assert sample["image"].dtype == torch.float32
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
            names={0: "class_0", 1: "class_1"},
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
        assert sample["image"].dtype == torch.float32
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
        image_size=(32, 32),
        channel_drop=None,
        num_channels="auto",
        normalize=NormalizeArgs(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        random_flip=None,
        color_jitter=None,
        scale_jitter=None,
        smallest_max_size=None,
        random_crop=None,
        bbox_params=BboxParams(format="yolo", label_fields=["class_labels"]),
    )
    return InstanceSegmentationTransform(transform_args)
