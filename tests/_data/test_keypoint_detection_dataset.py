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
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Union

import pytest
import yaml
from pydantic import Field, ValidationError, field_validator
from typing_extensions import Annotated

from lightly_train._commands import data_helpers as command_data_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.keypoint_detection_dataset import (
    COCOKeypointDetectionDataArgs,
    COCOSplitArgs,
    KeypointDetectionDataset,
    YOLOKeypointDetectionDataArgs,
)
from lightly_train._data.keypoint_helpers import KeypointSetArgs

from .. import helpers

# The default fixtures use three keypoints so that expected values stay readable, with
# the visibility cycling through visible, occluded and unlabeled.
NUM_KEYPOINTS = 3
EXPECTED_KEYPOINTS = [[0.3, 0.35], [0.35, 0.4], [0.0, 0.0]]
EXPECTED_VISIBILITY = [2, 1, 0]


def _decode(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "bboxes": json.loads(row["bboxes"]),
        "class_labels": json.loads(row["class_labels"]),
        "keypoints": json.loads(row["keypoints"]),
        "keypoint_visibility": json.loads(row["keypoint_visibility"]),
    }


def _round(value: Any, ndigits: int = 6) -> Any:
    """Rounds floats in a nested structure.

    pytest.approx does not support nested sequences, and the normalized coordinates
    carry the float noise of the pixel-to-[0, 1] division.
    """
    if isinstance(value, list):
        return [_round(item, ndigits) for item in value]
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def _yolo_data_args(tmp_path: Path, **kwargs: Any) -> YOLOKeypointDetectionDataArgs:
    defaults: Dict[str, Any] = dict(
        path=tmp_path,
        train="images/train",
        val="images/val",
        names={0: "person"},
        kpt_shape=[NUM_KEYPOINTS, 3],
    )
    defaults.update(kwargs)
    return YOLOKeypointDetectionDataArgs(**defaults)


def _coco_data_args(tmp_path: Path, **kwargs: Any) -> COCOKeypointDetectionDataArgs:
    defaults: Dict[str, Any] = dict(
        train=COCOSplitArgs(annotations=tmp_path / "train.json", images=Path("train")),
        val=COCOSplitArgs(annotations=tmp_path / "val.json", images=Path("val")),
    )
    defaults.update(kwargs)
    return COCOKeypointDetectionDataArgs(**defaults)


class TestYOLOKeypointDetectionDataArgs:
    def test_get_train_and_val_args(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        data_args = _yolo_data_args(tmp_path)

        train_args = data_args.get_train_args()
        assert train_args.image_dir == tmp_path / "images" / "train"
        assert train_args.label_dir == tmp_path / "labels" / "train"
        assert train_args.num_dims == 3
        assert train_args.keypoint_set.num_keypoints == NUM_KEYPOINTS

        val_args = data_args.get_val_args()
        assert val_args.image_dir == tmp_path / "images" / "val"
        assert val_args.label_dir == tmp_path / "labels" / "val"

    def test_included_classes(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(
            tmp_path, names={0: "a", 1: "b", 2: "c"}, ignore_classes={1}
        )
        assert data_args.included_classes == {0: "a", 2: "c"}
        assert data_args.num_included_classes == 2

    def test_validate_paths__missing_images_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="to include 'images' directory"):
            _yolo_data_args(tmp_path, train="train")

    def test_keypoint_set(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(
            tmp_path,
            flip_idx=[0, 2, 1],
            kpt_names={0: ["a", "b", "c"]},
            keypoints=KeypointSetArgs(
                num_keypoints=NUM_KEYPOINTS, sigmas=[0.1, 0.2, 0.3]
            ),
        )
        keypoint_set = data_args.keypoint_set
        assert keypoint_set.num_keypoints == NUM_KEYPOINTS
        assert keypoint_set.names == ["a", "b", "c"]
        assert keypoint_set.flip_idx == [0, 2, 1]
        assert keypoint_set.sigmas == [0.1, 0.2, 0.3]

    def test_invalid_kpt_shape_fails_at_construction(self, tmp_path: Path) -> None:
        # The keypoint set is resolved eagerly so that a bad config fails here rather
        # than much later when the labels are read.
        with pytest.raises(ValidationError, match="to be 2 for"):
            _yolo_data_args(tmp_path, kpt_shape=[3, 4])

    def test_conflicting_flip_idx_fails_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="Conflicting values for 'flip_idx'"):
            _yolo_data_args(
                tmp_path,
                flip_idx=[0, 2, 1],
                keypoints=KeypointSetArgs(
                    num_keypoints=NUM_KEYPOINTS, flip_idx=[0, 1, 2]
                ),
            )

    def test_kpt_names_of_ignored_class_may_differ(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(
            tmp_path,
            names={0: "a", 1: "b"},
            kpt_names={0: ["p", "q", "r"], 1: ["x", "y", "z"]},
            ignore_classes={1},
        )
        assert data_args.keypoint_set.names == ["p", "q", "r"]


class TestYOLOKeypointDetectionDatasetArgs:
    @pytest.mark.parametrize("split_first", [True, False])
    def test_list_image_info(self, tmp_path: Path, split_first: bool) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=split_first, num_files=2
        )
        data_args = _yolo_data_args(
            tmp_path,
            train="train/images" if split_first else "images/train",
            val="val/images" if split_first else "images/val",
        )

        for dataset_args in [data_args.get_train_args(), data_args.get_val_args()]:
            rows = [_decode(row) for row in dataset_args.list_image_info()]
            assert len(rows) == 2
            for row in rows:
                assert row["class_labels"] == [0]
                assert row["bboxes"] == [[0.375, 0.5, 0.25, 0.5]]
                # Already normalized by the YOLO format, so passed through unchanged.
                assert row["keypoints"] == [EXPECTED_KEYPOINTS]
                assert row["keypoint_visibility"] == [EXPECTED_VISIBILITY]

    def test_list_image_info__num_dims_2(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False, num_dims=2
        )
        data_args = _yolo_data_args(tmp_path, kpt_shape=[NUM_KEYPOINTS, 2])

        rows = [_decode(row) for row in data_args.get_train_args().list_image_info()]
        for row in rows:
            # Without a visibility channel every keypoint is labeled and visible, so
            # none of them sits at the origin.
            assert row["keypoint_visibility"] == [[2, 2, 2]]
            assert row["keypoints"] == [[[0.3, 0.35], [0.35, 0.4], [0.4, 0.45]]]

    def test_list_image_info__all_columns_are_str(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        data_args = _yolo_data_args(tmp_path)
        for row in data_args.get_train_args().list_image_info():
            assert set(row) == {
                "image_path",
                "bboxes",
                "class_labels",
                "keypoints",
                "keypoint_visibility",
            }
            assert all(isinstance(value, str) for value in row.values())

    def test_list_image_info__missing_label_file(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path,
            split_first=False,
            num_files=2,
            missing_label_indices=[0],
        )
        data_args = _yolo_data_args(tmp_path)
        rows = [_decode(row) for row in data_args.get_train_args().list_image_info()]
        assert len(rows) == 2
        # The images are not listed in a defined order, so compare the outcomes as a
        # multiset: the image without a label file has no instances, the other one has.
        assert sorted(len(row["class_labels"]) for row in rows) == [0, 1]
        empty_row = next(row for row in rows if not row["class_labels"])
        assert empty_row["keypoints"] == []
        assert empty_row["keypoint_visibility"] == []

    def test_list_image_info__skip_if_label_file_missing(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path,
            split_first=False,
            num_files=2,
            missing_label_indices=[0],
        )
        data_args = _yolo_data_args(tmp_path, skip_if_label_file_missing=True)
        rows = list(data_args.get_train_args().list_image_info())
        assert len(rows) == 1

    def test_list_image_info__empty_label_file(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False, num_files=2, empty_label_indices=[0]
        )
        data_args = _yolo_data_args(tmp_path)
        rows = [_decode(row) for row in data_args.get_train_args().list_image_info()]
        assert len(rows) == 2
        assert sorted(len(row["class_labels"]) for row in rows) == [0, 1]
        empty_row = next(row for row in rows if not row["class_labels"])
        assert empty_row["keypoints"] == []
        assert empty_row["keypoint_visibility"] == []

    def test_list_image_info__ignore_classes(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        # The fixture only writes class 0, so ignoring it drops every instance.
        data_args = _yolo_data_args(
            tmp_path, names={0: "a", 1: "b"}, ignore_classes={0}
        )
        rows = [_decode(row) for row in data_args.get_train_args().list_image_info()]
        assert all(row["class_labels"] == [] for row in rows)
        assert all(row["keypoints"] == [] for row in rows)

    def test_list_image_info__remaps_class_labels(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        # Class 5 is the only included class, so it maps to internal id 0.
        tmp_path_labels = tmp_path / "labels" / "train"
        for label_path in tmp_path_labels.glob("*.txt"):
            text = label_path.read_text()
            label_path.write_text("5" + text[1:])
        data_args = _yolo_data_args(
            tmp_path, names={3: "a", 5: "b"}, ignore_classes={3}
        )
        rows = [_decode(row) for row in data_args.get_train_args().list_image_info()]
        assert all(row["class_labels"] == [0] for row in rows)

    def test_list_image_info__min_keypoints(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        # The fixture instance has two labeled keypoints out of three.
        kept = _yolo_data_args(tmp_path, min_keypoints=2).get_train_args()
        dropped = _yolo_data_args(tmp_path, min_keypoints=3).get_train_args()
        assert all(
            _decode(row)["class_labels"] == [0] for row in kept.list_image_info()
        )
        assert all(
            _decode(row)["class_labels"] == [] for row in dropped.list_image_info()
        )

    def test_get_dataset_cls(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        dataset_args = _yolo_data_args(tmp_path).get_train_args()
        assert dataset_args.get_dataset_cls() is KeypointDetectionDataset


class TestCOCOKeypointDetectionDataArgs:
    def test_get_train_and_val_args(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        data_args = _coco_data_args(tmp_path)

        train_args = data_args.get_train_args()
        assert train_args.labels == tmp_path / "train.json"
        assert train_args.data_dir == Path("train")
        assert train_args.keypoint_set.num_keypoints == NUM_KEYPOINTS

        val_args = data_args.get_val_args()
        assert val_args.labels == tmp_path / "val.json"
        assert val_args.data_dir == Path("val")

    def test_included_classes(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path, classes={3: "cat", 7: "dog"}
        )
        data_args = _coco_data_args(tmp_path, ignore_classes={7})
        assert data_args.included_classes == {3: "cat"}
        assert data_args.num_included_classes == 1

    def test_keypoint_set_from_categories(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        keypoint_set = _coco_data_args(tmp_path).keypoint_set
        assert keypoint_set.num_keypoints == NUM_KEYPOINTS
        assert keypoint_set.names == ["keypoint_0", "keypoint_1", "keypoint_2"]
        # The fixture writes a one-indexed chain, as the COCO format requires.
        assert keypoint_set.skeleton == [[0, 1], [1, 2]]
        # The COCO format carries neither of these.
        assert keypoint_set.sigmas is None
        assert keypoint_set.flip_idx is None

    def test_keypoint_set__conflicting_categories(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            classes={0: "person", 1: "cat"},
            keypoint_names={0: ["a", "b", "c"], 1: ["x", "y"]},
        )
        with pytest.raises(ValueError, match="same keypoints"):
            _coco_data_args(tmp_path).keypoint_set

    def test_keypoint_set__conflicting_category_ignored(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            classes={0: "person", 1: "cat"},
            keypoint_names={0: ["a", "b", "c"], 1: ["x", "y"]},
        )
        data_args = _coco_data_args(tmp_path, ignore_classes={1})
        assert data_args.keypoint_set.names == ["a", "b", "c"]

    def test_keypoint_set__category_without_keypoints(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            classes={0: "person", 1: "ball"},
            keypoint_names={0: ["a", "b", "c"], 1: []},
        )
        # A file that mixes keypoint and non-keypoint categories is valid.
        assert _coco_data_args(tmp_path).keypoint_set.names == ["a", "b", "c"]

    def test_keypoint_set__no_category_with_keypoints(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path, classes={0: "ball"}, keypoint_names={0: []}
        )
        with pytest.raises(ValueError, match="Pass it explicitly"):
            _coco_data_args(tmp_path).keypoint_set

    def test_keypoint_set__user_spec_when_categories_have_none(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path, classes={0: "ball"}, keypoint_names={0: []}
        )
        keypoints = KeypointSetArgs(num_keypoints=NUM_KEYPOINTS, sigmas=[0.1, 0.2, 0.3])
        data_args = _coco_data_args(tmp_path, keypoints=keypoints)
        assert data_args.keypoint_set == keypoints

    def test_keypoint_set__user_spec_supplies_sigmas(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        data_args = _coco_data_args(
            tmp_path,
            keypoints=KeypointSetArgs(
                num_keypoints=NUM_KEYPOINTS, sigmas=[0.1, 0.2, 0.3], flip_idx=[0, 2, 1]
            ),
        )
        keypoint_set = data_args.keypoint_set
        assert keypoint_set.names == ["keypoint_0", "keypoint_1", "keypoint_2"]
        assert keypoint_set.sigmas == [0.1, 0.2, 0.3]
        assert keypoint_set.flip_idx == [0, 2, 1]

    def test_keypoint_set__user_spec_must_agree(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        data_args = _coco_data_args(
            tmp_path, keypoints=KeypointSetArgs(num_keypoints=NUM_KEYPOINTS + 1)
        )
        with pytest.raises(ValueError, match="Conflicting values for 'num_keypoints'"):
            data_args.keypoint_set


class TestCOCOKeypointDetectionDatasetArgs:
    def test_list_image_info(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path, num_files=2, height=128, width=128
        )
        data_args = _coco_data_args(tmp_path)

        for dataset_args in [data_args.get_train_args(), data_args.get_val_args()]:
            rows = [_decode(row) for row in dataset_args.list_image_info()]
            assert len(rows) == 2
            for row in rows:
                assert row["class_labels"] == [0]
                # bbox [10, 10, 30, 40] in pixels on a 128x128 image.
                assert _round(row["bboxes"]) == _round(
                    [[25 / 128, 30 / 128, 30 / 128, 40 / 128]]
                )
                assert _round(row["keypoints"]) == _round([EXPECTED_KEYPOINTS])
                assert row["keypoint_visibility"] == [EXPECTED_VISIBILITY]

    def test_list_image_info__all_columns_are_str(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        for row in _coco_data_args(tmp_path).get_train_args().list_image_info():
            assert set(row) == {
                "image_path",
                "bboxes",
                "class_labels",
                "keypoints",
                "keypoint_visibility",
            }
            assert all(isinstance(value, str) for value in row.values())

    def test_list_image_info__unlabeled_keypoints_are_zeroed(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        # An unlabeled keypoint that still carries a position.
                        "keypoints": [64, 64, 2, 32, 32, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    }
                ]
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert rows[0]["keypoints"] == [[[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]]]
        assert rows[0]["keypoint_visibility"] == [[2, 0, 0]]

    def test_list_image_info__coords_not_clipped(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [-32, 192, 2, 64, 64, 2, 0, 0, 0],
                        "num_keypoints": 2,
                    }
                ]
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        # Transforms need to know where an out-of-frame keypoint actually is.
        assert _round(rows[0]["keypoints"][0][0]) == [-0.25, 1.5]

    def test_list_image_info__num_keypoints_zero_kept_by_default(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [0] * (3 * NUM_KEYPOINTS),
                        "num_keypoints": 0,
                    }
                ]
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert rows[0]["class_labels"] == [0]
        assert rows[0]["keypoint_visibility"] == [[0, 0, 0]]

    def test_list_image_info__min_keypoints_drops_unlabeled_instances(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [0] * (3 * NUM_KEYPOINTS),
                        "num_keypoints": 0,
                    },
                    {
                        "category_id": 0,
                        "bbox": [20, 20, 30, 40],
                        "keypoints": [64, 64, 2, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    },
                ]
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path, min_keypoints=1)
            .get_train_args()
            .list_image_info()
        ]
        assert rows[0]["class_labels"] == [0]
        assert rows[0]["keypoint_visibility"] == [[2, 0, 0]]

    def test_list_image_info__recounts_num_keypoints(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 2, 32, 32, 2, 0, 0, 0],
                        # Wrong on purpose: the visibility flags say two.
                        "num_keypoints": 0,
                    }
                ]
            ],
        )
        # min_keypoints uses the recount, so the instance survives despite the stored 0.
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path, min_keypoints=2)
            .get_train_args()
            .list_image_info()
        ]
        assert rows[0]["class_labels"] == [0]

    def test_list_image_info__iscrowd_dropped_by_default(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 2, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                        "iscrowd": 1,
                    }
                ]
            ],
        )
        dropped = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert dropped[0]["class_labels"] == []

        kept = [
            _decode(row)
            for row in _coco_data_args(tmp_path, include_crowd=True)
            .get_train_args()
            .list_image_info()
        ]
        assert kept[0]["class_labels"] == [0]

    def test_list_image_info__annotation_without_keypoints(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            classes={0: "person", 1: "ball"},
            keypoint_names={0: ["a", "b", "c"], 1: []},
            annotations_per_image=[[{"category_id": 1, "bbox": [10, 10, 30, 40]}]],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        # No keypoints field means nothing is labeled, which is not an error.
        assert rows[0]["class_labels"] == [1]
        assert rows[0]["keypoints"] == [[[0.0, 0.0]] * NUM_KEYPOINTS]
        assert rows[0]["keypoint_visibility"] == [[0, 0, 0]]

    def test_list_image_info__wrong_keypoints_length_raises(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 2, 32, 32, 2],
                        "num_keypoints": 2,
                    }
                ]
            ],
        )
        with pytest.raises(ValueError, match="to have 9 values for 3 keypoints, got 6"):
            list(_coco_data_args(tmp_path).get_train_args().list_image_info())

    def test_list_image_info__bad_visibility_raises(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 5, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    }
                ]
            ],
        )
        with pytest.raises(ValueError, match="visibility to be one of"):
            list(_coco_data_args(tmp_path).get_train_args().list_image_info())

    @pytest.mark.parametrize(
        "annotation",
        [
            # No bbox at all.
            {
                "category_id": 0,
                "keypoints": [32, 32, 2, 96, 96, 2, 0, 0, 0],
                "num_keypoints": 2,
            },
            # A degenerate bbox is as useless as a missing one.
            {
                "category_id": 0,
                "bbox": [10, 10, 0, 40],
                "keypoints": [32, 32, 2, 96, 96, 2, 0, 0, 0],
                "num_keypoints": 2,
            },
        ],
    )
    def test_list_image_info__bbox_derived_from_keypoints(
        self, tmp_path: Path, annotation: Dict[str, Any]
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path, num_files=1, annotations_per_image=[[annotation]]
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        # Tight box around (0.25, 0.25) and (0.75, 0.75).
        assert _round(rows[0]["bboxes"]) == [[0.5, 0.5, 0.5, 0.5]]

    def test_list_image_info__no_bbox_and_no_keypoints_dropped(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            annotations_per_image=[
                [{"category_id": 0, "keypoints": [0] * 9, "num_keypoints": 0}]
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert rows[0]["class_labels"] == []

    def test_list_image_info__image_without_annotations(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=2,
            annotations_per_image=[
                [],
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 2, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    }
                ],
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert len(rows) == 2
        assert rows[0]["class_labels"] == []
        assert rows[1]["class_labels"] == [0]

    def test_list_image_info__skip_if_annotations_missing(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=2,
            annotations_per_image=[
                [],
                [
                    {
                        "category_id": 0,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 2, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    }
                ],
            ],
        )
        rows = list(
            _coco_data_args(tmp_path, skip_if_annotations_missing=True)
            .get_train_args()
            .list_image_info()
        )
        assert len(rows) == 1

    def test_list_image_info__ignore_classes(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path=tmp_path,
            num_files=1,
            classes={3: "cat", 7: "dog"},
            keypoint_names={3: ["a", "b", "c"], 7: ["a", "b", "c"]},
            annotations_per_image=[
                [
                    {
                        "category_id": 3,
                        "bbox": [10, 10, 30, 40],
                        "keypoints": [64, 64, 2, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    },
                    {
                        "category_id": 7,
                        "bbox": [20, 20, 30, 40],
                        "keypoints": [32, 32, 2, 0, 0, 0, 0, 0, 0],
                        "num_keypoints": 1,
                    },
                ]
            ],
        )
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path, ignore_classes={3})
            .get_train_args()
            .list_image_info()
        ]
        # Class 7 is the only included class, so it maps to internal id 0, and only its
        # keypoints survive the filter.
        assert rows[0]["class_labels"] == [0]
        assert _round(rows[0]["keypoints"]) == [[[0.25, 0.25], [0.0, 0.0], [0.0, 0.0]]]

    def test_list_image_info__missing_image_size_raises(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path, num_files=1)
        annotations_path = tmp_path / "train.json"
        labels = json.loads(annotations_path.read_text())
        del labels["images"][0]["width"]
        annotations_path.write_text(json.dumps(labels))
        # Normalization makes the image size load-bearing.
        with pytest.raises(KeyError, match="width"):
            list(_coco_data_args(tmp_path).get_train_args().list_image_info())

    def test_get_dataset_cls(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        dataset_args = _coco_data_args(tmp_path).get_train_args()
        assert dataset_args.get_dataset_cls() is KeypointDetectionDataset


class TestYOLOKeypointDetectionMmapHash:
    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(tmp_path)
        assert data_args.train_data_mmap_hash() == data_args.train_data_mmap_hash()
        assert data_args.val_data_mmap_hash() == data_args.val_data_mmap_hash()

    def test_train_and_val_hashes_differ(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(tmp_path)
        assert data_args.train_data_mmap_hash() != data_args.val_data_mmap_hash()

    @pytest.mark.parametrize(
        "changed",
        [
            {"kpt_shape": [4, 3]},
            {"flip_idx": [0, 2, 1]},
            {"keypoints": KeypointSetArgs(num_keypoints=3, sigmas=[0.1, 0.2, 0.3])},
            {"min_keypoints": 1},
        ],
    )
    def test_mmap_hash_changes(self, tmp_path: Path, changed: Dict[str, Any]) -> None:
        before = _yolo_data_args(tmp_path).train_data_mmap_hash()
        after = _yolo_data_args(tmp_path, **changed).train_data_mmap_hash()
        assert before != after

    def test_mmap_hash_is_stable_across_hash_seeds(self, tmp_path: Path) -> None:
        # A single-process determinism check cannot catch a set whose str() depends on
        # PYTHONHASHSEED, so compare two interpreters started with different seeds.
        script = textwrap.dedent(
            f"""
            from lightly_train._data.keypoint_detection_dataset import (
                YOLOKeypointDetectionDataArgs,
            )
            from lightly_train._data.keypoint_helpers import KeypointSetArgs

            data_args = YOLOKeypointDetectionDataArgs(
                path={str(tmp_path)!r},
                train="images/train",
                val="images/val",
                names={{0: "a", 1: "b", 2: "c"}},
                kpt_shape=[3, 3],
                flip_idx=[0, 2, 1],
                keypoints=KeypointSetArgs(num_keypoints=3, sigmas=[0.1, 0.2, 0.3]),
                ignore_classes={{2, 1}},
            )
            print(data_args.train_data_mmap_hash())
            """
        )
        hashes = []
        for seed in ("0", "1"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            )
            hashes.append(result.stdout)
        assert hashes[0] == hashes[1]


class TestCOCOKeypointDetectionMmapHash:
    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        data_args = _coco_data_args(tmp_path)
        assert data_args.train_data_mmap_hash() == data_args.train_data_mmap_hash()
        assert data_args.val_data_mmap_hash() == data_args.val_data_mmap_hash()

    def test_mmap_hash_changes_when_annotations_modified(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        data_args = _coco_data_args(tmp_path)
        before = data_args.train_data_mmap_hash()
        annotations_path = tmp_path / "train.json"
        stat = annotations_path.stat()
        os.utime(annotations_path, (stat.st_atime, stat.st_mtime + 1))
        assert data_args.train_data_mmap_hash() != before

    @pytest.mark.parametrize(
        "changed",
        [
            {"min_keypoints": 1},
            {"include_crowd": True},
            {"keypoints": KeypointSetArgs(num_keypoints=3, sigmas=[0.1, 0.2, 0.3])},
        ],
    )
    def test_mmap_hash_changes(self, tmp_path: Path, changed: Dict[str, Any]) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path=tmp_path)
        before = _coco_data_args(tmp_path).train_data_mmap_hash()
        after = _coco_data_args(tmp_path, **changed).train_data_mmap_hash()
        assert before != after


class _KeypointDetectionDataConfig(PydanticConfig):
    """Local stand-in for the future KeypointDetectionTrainTaskConfig.

    The data args ship unwired from train_task.py, since there is no keypoint model to
    train yet. This class still exercises the discriminated union and the YAML loading
    path, where a field missing from the data args would be dropped silently. It moves
    into train_task.py unchanged once a model lands.
    """

    data: Annotated[
        Union[YOLOKeypointDetectionDataArgs, COCOKeypointDetectionDataArgs],
        Field(discriminator="format"),
    ]

    @field_validator("data", mode="before")
    @classmethod
    def _load_yaml_if_path(cls, v: Any) -> Any:
        data_dict = command_data_helpers.load_data_yaml_if_path(
            v, cls.model_fields["data"].annotation
        )
        return command_data_helpers.set_default_data_format(data_dict, default="yolo")


class TestKeypointDetectionDataConfig:
    @staticmethod
    def _write_yaml(tmp_path: Path, data: Dict[str, Any]) -> Path:
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(data))
        return data_yaml

    @staticmethod
    def _validate(data: Any) -> _KeypointDetectionDataConfig:
        # Validate instead of constructing, the way
        # validate.pydantic_model_validate does for the real train task configs: the
        # field annotation is the resolved union, while the "before" validator accepts a
        # path or a dict.
        return _KeypointDetectionDataConfig.model_validate({"data": data})

    def test_pose_data_yaml_roundtrip(self, tmp_path: Path) -> None:
        # A pose data.yaml as it is written in practice, with the keypoint keys at the
        # top level. They must survive load_data_yaml_if_path's key filtering.
        data_yaml = self._write_yaml(
            tmp_path,
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": [3, 3],
                "flip_idx": [0, 2, 1],
                "kpt_names": {0: ["a", "b", "c"]},
                "names": {0: "person"},
            },
        )
        config = self._validate(data_yaml)
        assert isinstance(config.data, YOLOKeypointDetectionDataArgs)
        assert config.data.kpt_shape == [3, 3]
        assert config.data.flip_idx == [0, 2, 1]
        assert config.data.kpt_names == {0: ["a", "b", "c"]}
        assert config.data.names == {0: "person"}
        assert config.data.data_config_file == data_yaml
        assert config.data.keypoint_set.names == ["a", "b", "c"]

    def test_ignores_unknown_yaml_keys(self, tmp_path: Path) -> None:
        # Foreign keys such as "download" are common in pose data configs.
        data_yaml = self._write_yaml(
            tmp_path,
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": [3, 3],
                "names": {0: "person"},
                "download": "https://example.com/dataset.zip",
            },
        )
        config = self._validate(data_yaml)
        assert isinstance(config.data, YOLOKeypointDetectionDataArgs)

    def test_format_defaults_to_yolo(self, tmp_path: Path) -> None:
        config = self._validate(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": [3, 3],
                "names": {0: "person"},
            }
        )
        assert isinstance(config.data, YOLOKeypointDetectionDataArgs)
        assert config.data.format == "yolo"

    def test_coco_format_selects_coco_member(self, tmp_path: Path) -> None:
        config = self._validate(
            {
                "format": "coco",
                "train": {"annotations": str(tmp_path / "train.json")},
                "val": {"annotations": str(tmp_path / "val.json")},
            }
        )
        assert isinstance(config.data, COCOKeypointDetectionDataArgs)

    def test_extra_key_forbidden(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            self._validate(
                {
                    "path": str(tmp_path),
                    "train": "images/train",
                    "val": "images/val",
                    "kpt_shape": [3, 3],
                    "names": {0: "person"},
                    "not_a_field": 1,
                }
            )


class TestKeypointDetectionDataset:
    def test_getitem_not_implemented(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path=tmp_path, split_first=False
        )
        dataset_args = _yolo_data_args(tmp_path).get_train_args()
        image_info: List[Dict[str, str]] = list(dataset_args.list_image_info())
        dataset = KeypointDetectionDataset(
            dataset_args=dataset_args, image_info=image_info
        )
        assert len(dataset) == len(image_info)
        with pytest.raises(NotImplementedError, match="cannot be iterated yet"):
            dataset[0]
