#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from lightly_train._data.instance_segmentation_dataset import (
    COCOInstanceSegmentationDataArgs,
    COCOInstanceSegmentationDatasetArgs,
    COCOSplitArgs,
    YOLOInstanceSegmentationDataArgs,
    YOLOInstanceSegmentationDatasetArgs,
)

from .. import helpers
from ..helpers import create_coco_instance_segmentation_dataset

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
            assert json.loads(info["segments"]) == [[_POLYGON]]
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
            assert json.loads(info["segments"]) == [[_POLYGON]]
            assert json.loads(info["bboxes"]) == [_BBOX]
            assert json.loads(info["class_labels"]) == [0]


# Polygon in pixel coords (128x128 image), matching create_coco_instance_segmentation_dataset default.
_COCO_POLYGON_PX = [10.0, 10.0, 40.0, 10.0, 40.0, 50.0, 10.0, 50.0]
_COCO_POLYGON_NORM = [v / 128 for v in _COCO_POLYGON_PX]
# Bbox from annotation [x, y, w, h] = [10, 10, 30, 40] → normalized x_center, y_center, w, h.
_COCO_BBOX = [25 / 128, 30 / 128, 30 / 128, 40 / 128]
# Bbox derived from polygon (same rectangle, so same result).
_COCO_BBOX_FROM_POLYGON = [25 / 128, 30 / 128, 30 / 128, 40 / 128]


class TestCOCOInstanceSegmentationDataArgs:
    def test_get_train_args(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(tmp_path)
        args = COCOInstanceSegmentationDataArgs(
            train=COCOSplitArgs(
                annotations=tmp_path / "train.json", images=Path("train")
            ),
            val=COCOSplitArgs(annotations=tmp_path / "val.json", images=Path("val")),
        )

        train_args = args.get_train_args()

        assert train_args.labels == tmp_path / "train.json"
        assert train_args.data_dir == Path("train")

    def test_get_val_args(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(tmp_path)
        args = COCOInstanceSegmentationDataArgs(
            train=COCOSplitArgs(
                annotations=tmp_path / "train.json", images=Path("train")
            ),
            val=COCOSplitArgs(annotations=tmp_path / "val.json", images=Path("val")),
        )

        val_args = args.get_val_args()

        assert val_args.labels == tmp_path / "val.json"
        assert val_args.data_dir == Path("val")

    def test_included_classes(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(tmp_path, num_classes=2)
        args = COCOInstanceSegmentationDataArgs(
            train=COCOSplitArgs(annotations=tmp_path / "train.json"),
            val=COCOSplitArgs(annotations=tmp_path / "val.json"),
            ignore_classes={1},
        )

        assert args.included_classes == {0: "class_0"}
        assert args.num_included_classes == 1


class TestCOCOInstanceSegmentationDatasetArgs:
    def test_list_image_info(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(
            tmp_path=tmp_path,
            num_files=3,
            height=128,
            width=128,
            num_classes=2,
        )
        args = COCOInstanceSegmentationDatasetArgs(
            labels=tmp_path / "train.json",
            data_dir=Path("train"),
            classes={0: "class_0", 1: "class_1"},
            ignore_classes=None,
            skip_if_annotations_missing=False,
        )

        image_info = list(args.list_image_info())

        assert len(image_info) == 3
        for info in image_info:
            assert Path(info["image_path"]).exists()
            assert json.loads(info["segments"]) == [[_COCO_POLYGON_NORM]]
            assert json.loads(info["bboxes"]) == [_COCO_BBOX]
            assert json.loads(info["class_labels"]) == [0]

    def test_list_image_info__bbox_derived_from_polygon(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(
            tmp_path=tmp_path,
            num_files=1,
            height=128,
            width=128,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "segmentation": [_COCO_POLYGON_PX],
                        # no "bbox" key
                    }
                ]
            ],
        )
        args = COCOInstanceSegmentationDatasetArgs(
            labels=tmp_path / "train.json",
            data_dir=Path("train"),
            classes={0: "class_0"},
            ignore_classes=None,
            skip_if_annotations_missing=False,
        )

        image_info = list(args.list_image_info())

        assert len(image_info) == 1
        assert json.loads(image_info[0]["bboxes"]) == [_COCO_BBOX_FROM_POLYGON]

    def test_list_image_info__ignore_classes(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(
            tmp_path=tmp_path,
            num_files=2,
            height=128,
            width=128,
            num_classes=2,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "segmentation": [_COCO_POLYGON_PX],
                    },
                    {
                        "category_id": 1,
                        "bbox": [10, 10, 30, 40],
                        "segmentation": [_COCO_POLYGON_PX],
                    },
                ],
                [
                    {
                        "category_id": 1,
                        "bbox": [10, 10, 30, 40],
                        "segmentation": [_COCO_POLYGON_PX],
                    }
                ],
            ],
        )
        args = COCOInstanceSegmentationDatasetArgs(
            labels=tmp_path / "train.json",
            data_dir=Path("train"),
            classes={0: "class_0", 1: "class_1"},
            ignore_classes={1},
            skip_if_annotations_missing=False,
        )

        image_info = list(args.list_image_info())

        assert len(image_info) == 2
        assert json.loads(image_info[0]["class_labels"]) == [0]
        assert json.loads(image_info[1]["class_labels"]) == []

    def test_list_image_info__skip_if_annotations_missing(self, tmp_path: Path) -> None:
        create_coco_instance_segmentation_dataset(
            tmp_path=tmp_path,
            num_files=2,
            height=128,
            width=128,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "segmentation": [_COCO_POLYGON_PX],
                    }
                ],
                [],  # no annotations for second image
            ],
        )
        args = COCOInstanceSegmentationDatasetArgs(
            labels=tmp_path / "train.json",
            data_dir=Path("train"),
            classes={0: "class_0"},
            ignore_classes=None,
            skip_if_annotations_missing=True,
        )

        image_info = list(args.list_image_info())

        assert len(image_info) == 1

    @pytest.mark.skipif(
        sys.version_info < (3, 9), reason="pycocotools requires Python >= 3.9"
    )
    def test_list_image_info__mixed_polygon_and_rle(self, tmp_path: Path) -> None:
        """Test a single image with polygon, compressed RLE, and uncompressed RLE annotations."""
        if sys.version_info >= (3, 9):  # Needed for Mypy
            from pycocotools import mask as coco_mask

            height, width = 128, 128

            # Create a compressed RLE from a polygon using pycocotools.
            rle_list = coco_mask.frPyObjects(
                [[int(v) for v in _COCO_POLYGON_PX]], height, width
            )
            compressed_rle = coco_mask.merge(rle_list)
            # counts is bytes, convert to str for JSON.
            counts_raw = compressed_rle["counts"]
            counts_str = (
                counts_raw.decode("utf-8")
                if isinstance(counts_raw, bytes)
                else counts_raw
            )
            compressed_rle_annotation = {
                "counts": counts_str,
                "size": compressed_rle["size"],
            }

            # Create an uncompressed RLE (counts as a list of ints).
            binary_mask = coco_mask.decode(compressed_rle)
            flat = binary_mask.flatten(order="F")
            counts: list[int] = []
            current: int = 0
            count: int = 0
            for val in flat:
                if val == current:
                    count += 1
                else:
                    counts.append(count)
                    current = int(val)
                    count = 1
            counts.append(count)
            uncompressed_rle_annotation = {
                "counts": counts,
                "size": [height, width],
            }

            annotations_per_image = [
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "segmentation": [_COCO_POLYGON_PX],
                    },
                    {
                        "category_id": 0,
                        "segmentation": compressed_rle_annotation,
                    },
                    {
                        "category_id": 0,
                        "segmentation": uncompressed_rle_annotation,
                    },
                ]
            ]

            create_coco_instance_segmentation_dataset(
                tmp_path=tmp_path,
                num_files=1,
                height=height,
                width=width,
                num_classes=1,
                annotations_per_image=annotations_per_image,
            )
            args = COCOInstanceSegmentationDatasetArgs(
                labels=tmp_path / "train.json",
                data_dir=Path("train"),
                classes={0: "class_0"},
                ignore_classes=None,
                skip_if_annotations_missing=False,
            )

            image_info = list(args.list_image_info())

            assert len(image_info) == 1
            info = image_info[0]
            segments = json.loads(info["segments"])
            bboxes = json.loads(info["bboxes"])
            class_labels = json.loads(info["class_labels"])

            # Three annotations: one polygon, two RLE.
            assert len(segments) == 3
            assert len(bboxes) == 3
            assert len(class_labels) == 3

            # First segment is a polygon (list of lists).
            assert isinstance(segments[0], list)
            # Second and third segments are RLE dicts (compressed).
            assert isinstance(segments[1], dict)
            assert isinstance(segments[1]["counts"], str)
            assert isinstance(segments[2], dict)
            assert isinstance(segments[2]["counts"], str)

            # All bboxes should be close (they represent the same shape).
            for bbox in bboxes:
                assert len(bbox) == 4

    @pytest.mark.skipif(
        sys.version_info >= (3, 9), reason="Test only applies to Python 3.8"
    )
    def test_list_image_info__rle_fails_on_python38(self, tmp_path: Path) -> None:
        """RLE annotations should raise RuntimeError on Python < 3.9."""
        create_coco_instance_segmentation_dataset(
            tmp_path=tmp_path,
            num_files=1,
            height=128,
            width=128,
            num_classes=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "segmentation": {
                            "counts": [0, 5, 3],
                            "size": [128, 128],
                        },
                    }
                ]
            ],
        )
        args = COCOInstanceSegmentationDatasetArgs(
            labels=tmp_path / "train.json",
            data_dir=Path("train"),
            classes={0: "class_0"},
            ignore_classes=None,
            skip_if_annotations_missing=False,
        )

        with pytest.raises(RuntimeError, match="Python >= 3.9"):
            list(args.list_image_info())
