#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
from pathlib import Path

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)

from ..helpers import create_yolo_object_detection_dataset


class TestYOLOObjectDetectionDatasetArgs:
    def test_list_image_info__split_first(self, tmp_path: Path) -> None:
        create_yolo_object_detection_dataset(
            tmp_path=tmp_path, split_first=True, num_files=2
        )

        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0", 1: "class_1"},
        )

        train_image_info = list(args.get_train_args().list_image_info())
        val_image_info = list(args.get_val_args().list_image_info())

        assert len(train_image_info) == 2
        for info in train_image_info:
            assert Path(info["image_path"]).exists()
            bboxes = json.loads(info["bboxes"])
            class_labels = json.loads(info["class_labels"])
            assert len(bboxes) == 1
            assert len(class_labels) == 1
            assert bboxes[0] == [0.375, 0.5, 0.25, 0.5]
            assert class_labels[0] == 0

        assert len(val_image_info) == 2
        for info in val_image_info:
            assert Path(info["image_path"]).exists()
            bboxes = json.loads(info["bboxes"])
            class_labels = json.loads(info["class_labels"])
            assert len(bboxes) == 1
            assert len(class_labels) == 1
            assert bboxes[0] == [0.375, 0.5, 0.25, 0.5]
            assert class_labels[0] == 0

    def test_list_image_info__split_last(self, tmp_path: Path) -> None:
        create_yolo_object_detection_dataset(
            tmp_path=tmp_path, split_first=False, num_files=3
        )

        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train="images/train",
            val="images/val",
            names={7: "class_7", 0: "class_0"},
        )

        train_image_info = list(args.get_train_args().list_image_info())
        val_image_info = list(args.get_val_args().list_image_info())

        assert len(train_image_info) == 3
        for info in train_image_info:
            assert Path(info["image_path"]).exists()
            bboxes = json.loads(info["bboxes"])
            class_labels = json.loads(info["class_labels"])
            assert bboxes == [[0.375, 0.5, 0.25, 0.5]]
            assert class_labels == [1]

        assert len(val_image_info) == 3
        for info in val_image_info:
            assert Path(info["image_path"]).exists()
            bboxes = json.loads(info["bboxes"])
            class_labels = json.loads(info["class_labels"])
            assert bboxes == [[0.375, 0.5, 0.25, 0.5]]
            assert class_labels == [1]

    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        create_yolo_object_detection_dataset(
            tmp_path=tmp_path, split_first=True, num_files=2
        )
        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0", 1: "class_1"},
        )
        assert args.train_data_mmap_hash() == args.train_data_mmap_hash()
        assert args.val_data_mmap_hash() == args.val_data_mmap_hash()
