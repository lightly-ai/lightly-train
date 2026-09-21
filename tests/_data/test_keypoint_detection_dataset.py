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
    YOLOKeypointDetectionDataArgs,
)


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


def _write_coco_labels(tmp_path: Path) -> None:
    labels = {"categories": [{"id": 0, "name": "person", "keypoints": ["a", "b", "c"]}]}
    for split in ("train", "val"):
        (tmp_path / f"{split}.json").write_text(json.dumps(labels))


class TestYOLOKeypointDetectionDataArgs:
    def test_kpt_shape_accepts_yaml_list(self, tmp_path: Path) -> None:
        assert _yolo_data_args(tmp_path).kpt_shape == (3, 3)

    def test_native_keypoint_fields(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(
            tmp_path,
            flip_idx=[0, 2, 1],
            kpt_names={0: ["a", "b", "c"]},
            kpt_oks_sigmas=[0.1, 0.2, 0.3],
        )
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
    def test_invalid_metadata(
        self, tmp_path: Path, kwargs: Dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            _yolo_data_args(tmp_path, **kwargs)

    def test_hash_includes_num_dims(self, tmp_path: Path) -> None:
        hash_2d = _yolo_data_args(tmp_path, kpt_shape=[3, 2]).train_data_mmap_hash()
        hash_3d = _yolo_data_args(tmp_path, kpt_shape=[3, 3]).train_data_mmap_hash()
        assert hash_2d != hash_3d

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_args = _yolo_data_args(tmp_path, path="dataset")
        data_args.data_config_file = tmp_path / "configs" / "data.yaml"
        data_helpers.resolve_data_paths(data_args)
        assert data_args.path == (tmp_path / "configs" / "dataset").resolve()


class TestCOCOKeypointDetectionDataArgs:
    def test_reads_num_keypoints_from_categories(self, tmp_path: Path) -> None:
        _write_coco_labels(tmp_path)
        data_args = COCOKeypointDetectionDataArgs(
            train=COCOSplitArgs(annotations=tmp_path / "train.json"),
            val=COCOSplitArgs(annotations=tmp_path / "val.json"),
        )
        assert data_args.num_keypoints == 3

    def test_ignored_category_may_differ(self, tmp_path: Path) -> None:
        labels = {
            "categories": [
                {"id": 0, "name": "person", "keypoints": ["a", "b", "c"]},
                {"id": 1, "name": "animal", "keypoints": ["x", "y"]},
            ]
        }
        for split in ("train", "val"):
            (tmp_path / f"{split}.json").write_text(json.dumps(labels))
        data_args = COCOKeypointDetectionDataArgs(
            train=COCOSplitArgs(annotations=tmp_path / "train.json"),
            val=COCOSplitArgs(annotations=tmp_path / "val.json"),
            ignore_classes={1},
        )
        assert data_args.num_keypoints == 3


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
