#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Union

import pytest
import yaml
from pydantic import Field, ValidationError, field_validator
from typing_extensions import Annotated

from lightly_train._commands import data_helpers as command_data_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.keypoint_detection_dataset import (
    COCOKeypointDetectionDataArgs,
    COCOSplitArgs,
    YOLOKeypointDetectionDataArgs,
)
from lightly_train._data.keypoint_helpers import KeypointSetArgs

from .. import helpers

# The default fixtures use three keypoints so that expected values stay readable, with
# the visibility cycling through visible, occluded and unlabeled.
NUM_KEYPOINTS = 3
EXPECTED_KEYPOINTS = [[0.3, 0.35], [0.35, 0.4], [0.0, 0.0]]
EXPECTED_VISIBILITY = [2, 1, 0]


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


class TestCOCOKeypointDetectionDataArgs:
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
