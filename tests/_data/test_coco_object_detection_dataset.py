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
from pathlib import Path

from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
    SplitArgs,
)

from ..helpers import create_coco_object_detection_dataset


class TestCOCOObjectDetectionDatasetArgs:
    def test_list_image_info(self, tmp_path: Path) -> None:
        create_coco_object_detection_dataset(
            tmp_path=tmp_path,
            num_files=3,
            height=128,
            width=128,
            classes={3: "cat", 7: "dog"},
            annotations_per_image=[
                [],
                [{"category_id": 3, "bbox": [0, 32, 64, 64]}],
                [
                    {"category_id": 3, "bbox": [0, 64, 64, 64]},
                    {"category_id": 7, "bbox": [16, 0, 0, 128]},
                ],
            ],
        )

        args = COCOObjectDetectionDataArgs(
            train=SplitArgs(annotations=tmp_path / "train.json", images=Path("train")),
            val=SplitArgs(annotations=tmp_path / "val.json", images=Path("val")),
        )

        # class 3 -> internal 0, class 7 -> internal 1
        for dataset_args in [args.get_train_args(), args.get_val_args()]:
            image_info = list(dataset_args.list_image_info())
            assert len(image_info) == 3
            assert json.loads(image_info[0]["bboxes"]) == []
            assert json.loads(image_info[0]["class_labels"]) == []
            assert json.loads(image_info[1]["bboxes"]) == [[0.25, 0.5, 0.5, 0.5]]
            assert json.loads(image_info[1]["class_labels"]) == [0]
            assert json.loads(image_info[2]["bboxes"]) == [
                [0.25, 0.75, 0.5, 0.5],
                [0.125, 0.5, 0.0, 1.0],
            ]
            assert json.loads(image_info[2]["class_labels"]) == [0, 1]

    def _make_args(self, tmp_path: Path) -> COCOObjectDetectionDataArgs:
        create_coco_object_detection_dataset(tmp_path=tmp_path)
        return COCOObjectDetectionDataArgs(
            train=SplitArgs(annotations=tmp_path / "train.json", images=Path("train")),
            val=SplitArgs(annotations=tmp_path / "val.json", images=Path("val")),
        )

    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        args = self._make_args(tmp_path)
        assert args.train_data_mmap_hash() == args.train_data_mmap_hash()
        assert args.val_data_mmap_hash() == args.val_data_mmap_hash()

    def test_mmap_hash_changes_when_annotations_modified(self, tmp_path: Path) -> None:
        args = self._make_args(tmp_path)
        hash_before = args.train_data_mmap_hash()
        annotations_path = tmp_path / "train.json"
        st = annotations_path.stat()
        os.utime(annotations_path, (st.st_atime, st.st_mtime + 1))
        assert args.train_data_mmap_hash() != hash_before
