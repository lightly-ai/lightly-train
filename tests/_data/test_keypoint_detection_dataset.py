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
from typing import Any, Dict, Union

import pytest
import yaml
from pydantic import Field, ValidationError, field_validator
from typing_extensions import Annotated

from lightly_train._commands import data_helpers as command_data_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._data import data_helpers
from lightly_train._data.keypoint_detection_dataset import (
    COCOKeypointDetectionDataArgs,
    COCOSplitArgs,
    KeypointDetectionDataset,
    YOLOKeypointDetectionDataArgs,
)

from .. import helpers


def _decode(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "bboxes": json.loads(row["bboxes"]),
        "class_labels": json.loads(row["class_labels"]),
        "keypoints": json.loads(row["keypoints"]),
        "keypoint_visibility": json.loads(row["keypoint_visibility"]),
    }


def _yolo_data_args(tmp_path: Path, **kwargs: Any) -> YOLOKeypointDetectionDataArgs:
    defaults: Dict[str, Any] = {
        "path": tmp_path,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
        "kpt_shape": [3, 3],
    }
    defaults.update(kwargs)
    return YOLOKeypointDetectionDataArgs(**defaults)


def _coco_data_args(tmp_path: Path, **kwargs: Any) -> COCOKeypointDetectionDataArgs:
    defaults: Dict[str, Any] = {
        "train": COCOSplitArgs(annotations=tmp_path / "train.json", images="train"),
        "val": COCOSplitArgs(annotations=tmp_path / "val.json", images="val"),
    }
    defaults.update(kwargs)
    return COCOKeypointDetectionDataArgs(**defaults)


class TestYOLOKeypointDetectionDataArgs:
    def test_kpt_shape_accepts_yaml_list(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(tmp_path)
        assert data_args.kpt_shape == (3, 3)

    def test_kpt_metadata(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(
            tmp_path,
            flip_idx=[0, 2, 1],
            kpt_names={0: ["a", "b", "c"]},
            kpt_oks_sigmas=[0.1, 0.2, 0.3],
        )
        assert data_args.flip_idx == [0, 2, 1]
        assert data_args.kpt_names == {0: ["a", "b", "c"]}
        assert data_args.kpt_oks_sigmas == [0.1, 0.2, 0.3]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"kpt_shape": [3, 4]}, "to be 2 for"),
            ({"flip_idx": [0, 0, 2]}, "flip_idx"),
            ({"kpt_names": {0: ["a", "b"]}}, "kpt_names"),
            ({"kpt_oks_sigmas": [0.1, 0.2]}, "kpt_oks_sigmas"),
        ],
    )
    def test_invalid_keypoint_metadata(
        self, tmp_path: Path, kwargs: Dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            _yolo_data_args(tmp_path, **kwargs)

    def test_get_train_args(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(tmp_path, split_first=False)
        dataset_args = _yolo_data_args(tmp_path).get_train_args()
        assert dataset_args.num_keypoints == 3
        assert dataset_args.num_dims == 3
        assert dataset_args.image_dir == tmp_path / "images" / "train"

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(tmp_path, path="dataset")
        data_args.data_config_file = tmp_path / "configs" / "data.yaml"
        data_helpers.resolve_data_paths(data_args)
        assert data_args.path == (tmp_path / "configs" / "dataset").resolve()


class TestYOLOKeypointDetectionDatasetArgs:
    def test_list_image_info(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(tmp_path, split_first=False)
        rows = [
            _decode(row)
            for row in _yolo_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert len(rows) == 2
        assert rows[0]["class_labels"] == [0]
        assert rows[0]["keypoint_visibility"] == [[2, 1, 0]]

    def test_num_dims_2_marks_keypoints_visible(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path, split_first=False, num_dims=2
        )
        rows = [
            _decode(row)
            for row in _yolo_data_args(tmp_path, kpt_shape=[3, 2])
            .get_train_args()
            .list_image_info()
        ]
        assert rows[0]["keypoint_visibility"] == [[2, 2, 2]]

    def test_skip_if_label_file_missing(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(
            tmp_path, split_first=False, missing_label_indices=[0]
        )
        rows = list(
            _yolo_data_args(tmp_path, skip_if_label_file_missing=True)
            .get_train_args()
            .list_image_info()
        )
        assert len(rows) == 1

    def test_ignored_classes_are_removed(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(tmp_path, split_first=False)
        label_path = tmp_path / "labels" / "train" / "0.txt"
        label_path.write_text(label_path.read_text().replace("0 ", "1 "))
        rows = [
            _decode(row)
            for row in _yolo_data_args(
                tmp_path, names={0: "a", 1: "b"}, ignore_classes={1}
            )
            .get_train_args()
            .list_image_info()
        ]
        assert sorted(len(row["class_labels"]) for row in rows) == [0, 1]


class TestCOCOKeypointDetectionDataArgs:
    def test_get_train_args(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path)
        dataset_args = _coco_data_args(tmp_path).get_train_args()
        assert dataset_args.num_keypoints == 3
        assert dataset_args.data_dir == Path("train")

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_args = _coco_data_args(
            tmp_path,
            train=COCOSplitArgs(annotations="annotations/train.json", images="train"),
            val=COCOSplitArgs(annotations="annotations/val.json", images="val"),
        )
        data_args.data_config_file = tmp_path / "configs" / "data.yaml"
        data_helpers.resolve_data_paths(data_args)
        assert (
            data_args.train.annotations
            == (tmp_path / "configs" / "annotations/train.json").resolve()
        )

    def test_ignored_categories_may_have_different_keypoints(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_keypoint_detection_dataset(
            tmp_path,
            classes={0: "person", 1: "animal"},
            keypoint_names={0: ["a", "b", "c"], 1: ["x", "y"]},
        )
        for split in ("train", "val"):
            labels_path = tmp_path / f"{split}.json"
            labels = json.loads(labels_path.read_text())
            labels["annotations"].append(
                {
                    "id": 100,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [10, 10, 30, 40],
                    "keypoints": [10, 10, 2, 20, 20, 2],
                }
            )
            labels_path.write_text(json.dumps(labels))

        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path, ignore_classes={1})
            .get_train_args()
            .list_image_info()
        ]
        assert rows[0]["class_labels"] == [0]


class TestCOCOKeypointDetectionDatasetArgs:
    def test_list_image_info(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path)
        rows = [
            _decode(row)
            for row in _coco_data_args(tmp_path).get_train_args().list_image_info()
        ]
        assert len(rows) == 2
        assert rows[0]["bboxes"] == [[0.1953125, 0.234375, 0.234375, 0.3125]]

    def test_missing_annotations_are_skipped(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path)
        labels_path = tmp_path / "train.json"
        labels = json.loads(labels_path.read_text())
        labels["annotations"] = []
        labels_path.write_text(json.dumps(labels))
        assert (
            list(
                _coco_data_args(tmp_path, skip_if_annotations_missing=True)
                .get_train_args()
                .list_image_info()
            )
            == []
        )


class TestMmapHashes:
    def test_yolo_hash_changes_with_num_dims(self, tmp_path: Path) -> None:
        hash_2d = _yolo_data_args(tmp_path, kpt_shape=[3, 2]).train_data_mmap_hash()
        hash_3d = _yolo_data_args(tmp_path, kpt_shape=[3, 3]).train_data_mmap_hash()
        assert hash_2d != hash_3d

    def test_yolo_hash_changes_with_pose_metadata(self, tmp_path: Path) -> None:
        before = _yolo_data_args(tmp_path).train_data_mmap_hash()
        after = _yolo_data_args(
            tmp_path, kpt_oks_sigmas=[0.1, 0.2, 0.3]
        ).train_data_mmap_hash()
        assert before != after

    def test_coco_hash_is_deterministic(self, tmp_path: Path) -> None:
        helpers.create_coco_keypoint_detection_dataset(tmp_path)
        data_args = _coco_data_args(tmp_path)
        assert data_args.train_data_mmap_hash() == data_args.train_data_mmap_hash()


class _KeypointDetectionDataConfig(PydanticConfig):
    data: Annotated[
        Union[YOLOKeypointDetectionDataArgs, COCOKeypointDetectionDataArgs],
        Field(discriminator="format"),
    ]

    @field_validator("data", mode="before")
    @classmethod
    def load_yaml_if_path(cls, v: Any) -> Any:
        data_dict = command_data_helpers.load_data_yaml_if_path(
            v, cls.model_fields["data"].annotation
        )
        return command_data_helpers.set_default_data_format(data_dict, default="yolo")


class TestKeypointDetectionDataConfig:
    def test_yolo_yaml_roundtrip(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "kpt_shape": [3, 3],
                    "kpt_oks_sigmas": [0.1, 0.2, 0.3],
                    "names": {0: "person"},
                }
            )
        )
        config = _KeypointDetectionDataConfig.model_validate({"data": data_yaml})
        assert isinstance(config.data, YOLOKeypointDetectionDataArgs)
        assert config.data.kpt_shape == (3, 3)
        assert config.data.kpt_oks_sigmas == [0.1, 0.2, 0.3]


class TestKeypointDetectionDataset:
    def test_getitem_not_implemented(self, tmp_path: Path) -> None:
        helpers.create_yolo_keypoint_detection_dataset(tmp_path, split_first=False)
        dataset_args = _yolo_data_args(tmp_path).get_train_args()
        image_info = list(dataset_args.list_image_info())
        dataset = KeypointDetectionDataset(
            dataset_args=dataset_args, image_info=image_info
        )
        with pytest.raises(NotImplementedError, match="cannot be iterated yet"):
            dataset[0]
