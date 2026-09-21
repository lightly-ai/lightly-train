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
from lightly_train._data.keypoint_helpers import KeypointSetArgs
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train.types import PathLike


class YOLOKeypointDetectionDataArgs(TaskDataArgs):
    """Data arguments for a YOLO-format keypoint detection dataset.

    Labels are ``.txt`` files in a ``labels`` directory mirroring ``images``. One line
    per instance: ``class_id x_center y_center width height`` followed by
    ``kpt_shape[0]`` keypoints of ``kpt_shape[1]`` values each, normalized to [0, 1].

    Attributes:
        kpt_shape:
            ``[num_keypoints, num_dims]``. ``num_dims`` is 2 for (x, y) or 3 for
            (x, y, visibility). With 2 every keypoint is read as visible; the format
            cannot express an unlabeled one.
        flip_idx:
            Keypoint permutation for a horizontal flip.
        kpt_names:
            Keypoint names per class. One keypoint set is used for all classes, so all
            included classes must declare the same names.
        keypoints:
            Keypoint set fields the dataset config has no place for, above all the OKS
            ``sigmas`` and the ``skeleton``. Values given here and in the fields above
            must agree.
        min_keypoints:
            Drop instances with fewer labeled keypoints. Default 0 keeps everything the
            label file contains.
    """

    format: Literal["yolo"] = "yolo"
    path: PathLike
    train: PathLike
    val: PathLike
    # Accepted for compatibility with YOLO data configs. Task training currently
    # consumes only train and val splits.
    test: PathLike | None = None
    # "names" instead of "classes" to match YOLO convention.
    names: dict[int, str]
    kpt_shape: list[int]
    flip_idx: list[int] | None = None
    kpt_names: dict[int, list[str]] | None = None
    keypoints: KeypointSetArgs | None = None
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False
    min_keypoints: int = 0

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: PathLike) -> Path:
        v = Path(v)
        if "images" not in v.parts:
            raise ValueError(f"Expected path to include 'images' directory, got {v}.")
        return v

    @model_validator(mode="after")
    def validate_keypoint_set(self) -> Self:
        # Resolve at construction so a bad keypoint config fails here, not much later
        # when the labels are read. Not stored: assigning to self would re-trigger
        # validation because of validate_assignment=True.
        self._resolve_keypoint_set()
        return self

    def _resolve_keypoint_set(self) -> KeypointSetArgs:
        return keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=self.kpt_shape,
            flip_idx=self.flip_idx,
            kpt_names=self.kpt_names,
            keypoints=self.keypoints,
            included_class_ids=self.included_classes.keys(),
        )

    @functools.cached_property
    def keypoint_set(self) -> KeypointSetArgs:
        """The keypoint set, combining the dataset config and the 'keypoints' arg."""
        return self._resolve_keypoint_set()

    def resolve_data_paths(self, base_dir: Path) -> None:
        self.path = data_helpers.resolve_path(self.path, base_dir=base_dir)

    def train_data_mmap_hash(self) -> str:
        return str(
            (
                (Path(self.path) / self.train).resolve(),
                self.names,
                self.keypoint_set.model_dump_json(),
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_label_file_missing,
                self.min_keypoints,
            )
        )

    def val_data_mmap_hash(self) -> str:
        return str(
            (
                (Path(self.path) / self.val).resolve(),
                self.names,
                self.keypoint_set.model_dump_json(),
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_label_file_missing,
                self.min_keypoints,
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

    The keypoint set comes from the ``categories`` of the training annotations, which
    declare names and skeleton. The format declares neither OKS sigmas nor flip pairs;
    pass those via ``keypoints``.

    Attributes:
        keypoints:
            Keypoint set fields the annotations do not carry, above all the OKS
            ``sigmas`` and the ``flip_idx``. Values given here and in the annotations
            must agree. Required if no category declares keypoints.
        include_crowd:
            Keep ``iscrowd == 1`` annotations. They describe a group rather than one
            instance and carry no usable keypoints, so they are dropped by default.
        min_keypoints:
            Drop instances with fewer labeled keypoints. Default 0 keeps everything the
            annotations contain, including the ``num_keypoints == 0`` annotations that
            COCO keypoint files are full of.
    """

    format: Literal["coco"] = "coco"
    train: COCOSplitArgs
    val: COCOSplitArgs
    keypoints: KeypointSetArgs | None = None
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_annotations_missing: bool = False
    include_crowd: bool = False
    min_keypoints: int = 0

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
    def keypoint_set(self) -> KeypointSetArgs:
        """The keypoint set, combining the annotations and the 'keypoints' arg."""
        return keypoint_helpers.resolve_coco_keypoint_set(
            categories=self._categories,
            included_class_ids=self.included_classes.keys(),
            keypoints=self.keypoints,
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
                self.keypoint_set.model_dump_json(),
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_annotations_missing,
                self.include_crowd,
                self.min_keypoints,
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
