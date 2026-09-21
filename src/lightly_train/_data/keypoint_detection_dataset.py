#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Literal

import pydantic
from pydantic import Field, model_validator
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import (
    data_helpers,
    file_helpers,
    keypoint_helpers,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train.types import PathLike


class YOLOKeypointDetectionDataArgs(TaskDataArgs):
    """Data arguments for a YOLO-format keypoint detection dataset.

    Labels are ``.txt`` files in a ``labels`` directory mirroring ``images``. One line
    per instance: ``class_id x_center y_center width height`` followed by
    ``kpt_shape[0]`` keypoints of ``kpt_shape[1]`` values each, normalized to [0, 1].
    """

    format: Literal["yolo"] = "yolo"
    path: PathLike
    train: PathLike
    val: PathLike
    test: PathLike | None = None
    """Accepted for compatibility with YOLO data configs.

    Task training consumes only train and val.
    """
    names: dict[int, str]
    kpt_shape: tuple[int, int] = Field(strict=False)
    """The keypoint count and dimensionality.

    ``num_dims`` is 2 for (x, y) or 3 for (x, y, visibility).
    """
    flip_idx: list[int] | None = None
    """Optional keypoint mapping for horizontal flips.

    Horizontal flips must be disabled when this is omitted.
    """
    kpt_names: dict[int, list[str]] | None = None
    kpt_oks_sigmas: list[float] | None = Field(default=None, strict=False)
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: PathLike) -> Path:
        v = Path(v)
        if "images" not in v.parts:
            raise ValueError(f"Expected path to include 'images' directory, got {v}.")
        return v

    @model_validator(mode="after")
    def validate_keypoint_config(self) -> Self:
        num_keypoints, _ = keypoint_helpers.validate_kpt_shape(self.kpt_shape)
        keypoint_helpers.validate_flip_idx(self.flip_idx, num_keypoints)
        keypoint_helpers.validate_kpt_names(self.kpt_names, num_keypoints)
        keypoint_helpers.validate_kpt_oks_sigmas(self.kpt_oks_sigmas, num_keypoints)
        return self

    def resolve_data_paths(self, base_dir: Path) -> None:
        self.path = data_helpers.resolve_path(self.path, base_dir=base_dir)

    def train_data_mmap_hash(self) -> str:
        return str(
            (
                (Path(self.path) / self.train).resolve(),
                self.names,
                self.kpt_shape,
                self.flip_idx,
                self.kpt_names,
                self.kpt_oks_sigmas,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_label_file_missing,
            )
        )

    def val_data_mmap_hash(self) -> str:
        return str(
            (
                (Path(self.path) / self.val).resolve(),
                self.names,
                self.kpt_shape,
                self.flip_idx,
                self.kpt_names,
                self.kpt_oks_sigmas,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_label_file_missing,
            )
        )

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        ignore_classes = set() if self.ignore_classes is None else self.ignore_classes
        return {
            class_id: class_name
            for class_id, class_name in self.names.items()
            if class_id not in ignore_classes
        }

    @property
    def num_included_classes(self) -> int:
        return len(self.included_classes)


class COCOSplitArgs(PydanticConfig):
    annotations: PathLike
    images: PathLike | None = None


class COCOKeypointDetectionDataArgs(TaskDataArgs):
    """Data arguments for a COCO-format keypoint detection dataset.

    Labels are COCO JSON annotation files. Images resolve relative to the annotation
    file's parent directory, optionally under ``images``.

    The number of keypoints comes from the train annotations' ``categories``.
    """

    format: Literal["coco"] = "coco"
    train: COCOSplitArgs
    val: COCOSplitArgs
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_annotations_missing: bool = False
    include_crowd: bool = False

    def resolve_data_paths(self, base_dir: Path) -> None:
        self.train.annotations = data_helpers.resolve_path(
            self.train.annotations, base_dir=base_dir
        )
        self.val.annotations = data_helpers.resolve_path(
            self.val.annotations, base_dir=base_dir
        )
        if self.train.images is not None:
            train_images = Path(self.train.images)
            self.train.images = (
                train_images.resolve() if train_images.is_absolute() else train_images
            )
        if self.val.images is not None:
            val_images = Path(self.val.images)
            self.val.images = (
                val_images.resolve() if val_images.is_absolute() else val_images
            )

    @functools.cached_property
    def _categories(self) -> list[dict[str, Any]]:
        """Reads and caches the categories from the train labels file.

        Always the training labels, so train and val share the class-to-internal-id
        mapping and the keypoint set. Cached so the file is read once for both.
        """
        with open(self.train.annotations) as f:
            categories = json.load(f).get("categories", [])
        return [dict(category) for category in categories]

    @functools.cached_property
    def _classes(self) -> dict[int, str]:
        return {category["id"]: category["name"] for category in self._categories}

    @functools.cached_property
    def num_keypoints(self) -> int:
        """Returns the number of keypoints in the included categories."""
        return keypoint_helpers.get_coco_num_keypoints(
            categories=self._categories,
            included_class_ids=self.included_classes.keys(),
        )

    def train_data_mmap_hash(self) -> str:
        return self._data_mmap_hash(split=self.train)

    def val_data_mmap_hash(self) -> str:
        return self._data_mmap_hash(split=self.val)

    def _data_mmap_hash(self, split: COCOSplitArgs) -> str:
        annotations_path = Path(split.annotations).resolve()
        images_dir = file_helpers.resolve_coco_images_dir(
            annotations_path, split.images
        )
        return str(
            (
                annotations_path,
                annotations_path.stat().st_mtime,
                images_dir,
                self.num_keypoints,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_annotations_missing,
                self.include_crowd,
            )
        )

    @property
    def included_classes(self) -> dict[int, str]:
        """Returns included classes."""
        ignore_classes = set() if self.ignore_classes is None else self.ignore_classes
        return {
            class_id: class_name
            for class_id, class_name in self._classes.items()
            if class_id not in ignore_classes
        }

    @property
    def num_included_classes(self) -> int:
        return len(self.included_classes)
