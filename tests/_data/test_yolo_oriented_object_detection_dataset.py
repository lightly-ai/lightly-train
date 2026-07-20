#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
from albumentations import BboxParams
from lightning_utilities.core.imports import RequirementCache

from lightly_train._data.yolo_oriented_object_detection_dataset import (
    YOLOOrientedObjectDetectionDataArgs,
    YOLOOrientedObjectDetectionDataset,
)
from lightly_train._transforms.oriented_object_detection_transform import (
    OrientedObjectDetectionTransform,
    OrientedObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
)
from lightly_train.types import ImageSizeTuple

from ..helpers import create_yolo_oriented_object_detection_dataset

if not RequirementCache("torchvision>=0.23"):
    pytest.skip(
        "torchvision too old for oriented bounding box suport",
        allow_module_level=True,
    )


class DummyTransformArgs(OrientedObjectDetectionTransformArgs):
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = 3
    photometric_distort: RandomPhotometricDistortArgs | None = None
    random_zoom_out: RandomZoomOutArgs | None = None
    random_iou_crop: RandomIoUCropArgs | None = None
    random_flip: RandomFlipArgs | None = None
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple = (32, 32)
    scale_jitter: ScaleJitterArgs | None = None
    resize: ResizeArgs | None = None
    normalize: NormalizeArgs | Literal["auto"] | None = None
    bbox_params: BboxParams = BboxParams(
        format="yolo",
        label_fields=["class_labels"],
    )


class TestYoloOrientedObjectDetectionDataset:
    def test__split_first(self, tmp_path: Path) -> None:
        create_yolo_oriented_object_detection_dataset(
            tmp_path=tmp_path, split_first=True
        )

        args = YOLOOrientedObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0", 1: "class_1"},
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=train_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "train/images/0.png"),
                    "label_path": str(tmp_path / "train/labels/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "train/images/1.png"),
                    "label_path": str(tmp_path / "train/labels/1.txt"),
                },
            ],
        )

        val_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=val_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "val/images/0.png"),
                    "label_path": str(tmp_path / "val/labels/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "val/images/1.png"),
                    "label_path": str(tmp_path / "val/labels/1.txt"),
                },
            ],
        )

        sample = train_dataset[0]
        assert sample["image"].dtype == torch.float32
        # Oriented bboxes have 5 values: (cx, cy, w, h, angle)
        assert sample["bboxes"].shape == (1, 5)
        assert sample["classes"].shape == (1,)

        sample = val_dataset[0]
        assert sample["image"].dtype == torch.float32
        assert sample["bboxes"].shape == (1, 5)
        assert sample["classes"].shape == (1,)

    def test__split_last(self, tmp_path: Path) -> None:
        create_yolo_oriented_object_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )

        args = YOLOOrientedObjectDetectionDataArgs(
            path=tmp_path,
            train="images/train",
            val="images/val",
            names={0: "class_0", 1: "class_1"},
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=train_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "images/train/0.png"),
                    "label_path": str(tmp_path / "labels/train/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "images/train/1.png"),
                    "label_path": str(tmp_path / "labels/train/1.txt"),
                },
            ],
        )

        val_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=val_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "images/val/0.png"),
                    "label_path": str(tmp_path / "labels/val/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "images/val/1.png"),
                    "label_path": str(tmp_path / "labels/val/1.txt"),
                },
            ],
        )

        sample = train_dataset[0]
        assert sample["image"].dtype == torch.float32
        assert sample["bboxes"].shape == (1, 5)
        assert sample["classes"].shape == (1,)

        sample = val_dataset[0]
        assert sample["image"].dtype == torch.float32
        assert sample["bboxes"].shape == (1, 5)

    def test__get_item__internal_class_ids(self, tmp_path: Path) -> None:
        create_yolo_oriented_object_detection_dataset(
            tmp_path=tmp_path, split_first=True
        )

        args = YOLOOrientedObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0", 2: "class_2"},
        )
        expected_mapping = {0: 0, 2: 1}

        train_args = args.get_train_args()
        train_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=train_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "train/images/0.png"),
                    "label_path": str(tmp_path / "train/labels/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "train/images/1.png"),
                    "label_path": str(tmp_path / "train/labels/1.txt"),
                },
            ],
        )

        assert train_dataset.class_id_to_internal_class_id == expected_mapping

    def test__getitem__no_label_file(self, tmp_path: Path) -> None:
        create_yolo_oriented_object_detection_dataset(
            tmp_path=tmp_path, split_first=True, missing_label_indices=[0]
        )

        args = YOLOOrientedObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0"},
        )

        train_args = args.get_train_args()
        train_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=train_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "train/images/0.png"),
                    "label_path": str(tmp_path / "train/labels/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "train/images/1.png"),
                    "label_path": str(tmp_path / "train/labels/1.txt"),
                },
            ],
        )

        # Should handle missing label file without error
        sample = train_dataset[0]
        assert sample["image"].dtype == torch.float32
        assert sample["bboxes"].shape == (0, 5)
        assert sample["classes"].shape == (0,)

    def test__getitem__empty_label_file(self, tmp_path: Path) -> None:
        create_yolo_oriented_object_detection_dataset(
            tmp_path=tmp_path, split_first=True, empty_label_indices=[0]
        )

        args = YOLOOrientedObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0"},
        )

        train_args = args.get_train_args()
        train_dataset = YOLOOrientedObjectDetectionDataset(
            dataset_args=train_args,
            transform=OrientedObjectDetectionTransform(DummyTransformArgs()),
            image_info=[
                {
                    "image_path": str(tmp_path / "train/images/0.png"),
                    "label_path": str(tmp_path / "train/labels/0.txt"),
                },
                {
                    "image_path": str(tmp_path / "train/images/1.png"),
                    "label_path": str(tmp_path / "train/labels/1.txt"),
                },
            ],
        )

        # Should handle empty label file without error
        sample = train_dataset[0]
        assert sample["image"].dtype == torch.float32
        assert sample["bboxes"].shape == (0, 5)
        assert sample["classes"].shape == (0,)


class TestYOLOOrientedObjectDetectionMmapHash:
    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        create_yolo_oriented_object_detection_dataset(
            tmp_path=tmp_path, split_first=True
        )
        args = YOLOOrientedObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0", 1: "class_1"},
        )
        assert args.train_data_mmap_hash() == args.train_data_mmap_hash()
        assert args.val_data_mmap_hash() == args.val_data_mmap_hash()
