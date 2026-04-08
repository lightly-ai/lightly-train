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

import pytest

from lightly_train._data.instance_segmentation_dataset import (
    YOLOInstanceSegmentationDataArgs,
    YOLOInstanceSegmentationDatasetArgs,
)

from .. import helpers

_POLYGON = [0.30, 0.30, 0.45, 0.27, 0.49, 0.50, 0.44, 0.70, 0.31, 0.73, 0.26, 0.50]
# Bbox is (x_center, y_center, width, height) derived from _POLYGON.
_BBOX = [
    (0.26 + 0.49) / 2,  # x_center
    (0.27 + 0.73) / 2,  # y_center
    0.49 - 0.26,  # width
    0.73 - 0.27,  # height
]

_CLASSES = {0: "class_0", 1: "class_1"}


class TestYOLOInstanceSegmentationDataArgs:
    def test_get_train_args(self, tmp_path: Path) -> None:
        # Arrange
        args = YOLOInstanceSegmentationDataArgs(
            path=tmp_path, train="train/images", val="val/images", names=_CLASSES
        )

        # Act
        train_args = args.get_train_args()

        # Assert
        assert train_args.image_dir == tmp_path / "train" / "images"
        assert train_args.label_dir == tmp_path / "train" / "labels"

    def test_get_val_args(self, tmp_path: Path) -> None:
        # Arrange
        args = YOLOInstanceSegmentationDataArgs(
            path=tmp_path, train="images/train", val="images/val", names=_CLASSES
        )

        # Act
        val_args = args.get_val_args()

        # Assert
        assert val_args.image_dir == tmp_path / "images" / "val"
        assert val_args.label_dir == tmp_path / "labels" / "val"

    def test_included_classes(self, tmp_path: Path) -> None:
        # Arrange
        args = YOLOInstanceSegmentationDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names=_CLASSES,
            ignore_classes={1},
        )

        # Act / Assert
        assert args.included_classes == {0: "class_0"}
        assert args.num_included_classes == 1

    def test_validate_paths__missing_images_dir(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            YOLOInstanceSegmentationDataArgs(
                path=tmp_path,
                train="train/labels",
                val="val/labels",
                names=_CLASSES,
            )


class TestYOLOInstanceSegmentationDatasetArgs:
    def test_list_image_info__split_first(self, tmp_path: Path) -> None:
        # Arrange
        helpers.create_yolo_instance_segmentation_dataset(
            tmp_path=tmp_path, split_first=True, num_files=2
        )
        args = YOLOInstanceSegmentationDatasetArgs(
            image_dir=tmp_path / "train" / "images",
            label_dir=tmp_path / "train" / "labels",
            classes=_CLASSES,
            ignore_classes=None,
            skip_if_label_file_missing=False,
        )

        # Act
        image_info = list(args.list_image_info())

        # Assert
        assert len(image_info) == 2
        for info in image_info:
            assert Path(info["image_path"]).exists()
            assert json.loads(info["polygons"]) == [_POLYGON]
            assert json.loads(info["bboxes"]) == [_BBOX]
            assert json.loads(info["class_labels"]) == [0]

    def test_list_image_info__split_last(self, tmp_path: Path) -> None:
        # Arrange
        helpers.create_yolo_instance_segmentation_dataset(
            tmp_path=tmp_path, split_first=False, num_files=3
        )
        args = YOLOInstanceSegmentationDatasetArgs(
            image_dir=tmp_path / "images" / "train",
            label_dir=tmp_path / "labels" / "train",
            classes=_CLASSES,
            ignore_classes=None,
            skip_if_label_file_missing=False,
        )

        # Act
        image_info = list(args.list_image_info())

        # Assert
        assert len(image_info) == 3
        for info in image_info:
            assert Path(info["image_path"]).exists()
            assert json.loads(info["polygons"]) == [_POLYGON]
            assert json.loads(info["bboxes"]) == [_BBOX]
            assert json.loads(info["class_labels"]) == [0]
