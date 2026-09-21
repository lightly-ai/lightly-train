#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import functools
import itertools
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
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
    label_helpers,
    yolo_helpers,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train.types import KeypointDetectionDatasetItem, PathLike

logger = logging.getLogger(__name__)


class KeypointDetectionDataset(TaskDataset):
    """Dataset for keypoint detection.

    Reading labels works. Turning them into training samples does not: there is no
    keypoint detection model and no keypoint transform yet. Today the useful part is
    ``list_image_info`` on the dataset args.
    """

    dataset_args: (  # type: ignore[assignment]
        COCOKeypointDetectionDatasetArgs | YOLOKeypointDetectionDatasetArgs
    )

    def __getitem__(self, index: int) -> KeypointDetectionDatasetItem:
        raise NotImplementedError(
            "Keypoint detection datasets cannot be iterated yet. The keypoint "
            "transforms and task model are not implemented."
        )


def _filter_instances(
    keep: list[bool],
    bboxes: list[list[float]],
    class_labels: list[int],
    keypoints: list[list[list[float]]],
    visibility: list[list[int]],
) -> tuple[list[list[float]], list[int], list[list[list[float]]], list[list[int]]]:
    """Applies the same mask to all four per-instance lists.

    They must stay in lockstep. A misalignment here only surfaces during training.
    """
    return (
        list(itertools.compress(bboxes, keep)),
        list(itertools.compress(class_labels, keep)),
        list(itertools.compress(keypoints, keep)),
        list(itertools.compress(visibility, keep)),
    )


def _image_info(
    image_path: Path,
    bboxes: list[list[float]],
    class_labels: list[int],
    keypoints: list[list[list[float]]],
    visibility: list[list[int]],
    num_keypoints: int,
) -> dict[str, str]:
    """Serializes one image's labels into the flat string columns of the mmap."""
    assert len(bboxes) == len(class_labels) == len(keypoints) == len(visibility)
    assert all(len(instance) == num_keypoints for instance in keypoints)
    assert all(len(instance) == num_keypoints for instance in visibility)
    return {
        "image_path": str(image_path),
        "bboxes": json.dumps(bboxes),
        "class_labels": json.dumps(class_labels),
        "keypoints": json.dumps(keypoints),
        "keypoint_visibility": json.dumps(visibility),
    }


def _count_labeled(visibility: list[int]) -> int:
    return sum(1 for vis in visibility if vis != keypoint_helpers.VISIBILITY_UNLABELED)


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

    def _get_args(
        self, mode: Literal["train", "val"]
    ) -> YOLOKeypointDetectionDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode=mode,
        )
        assert image_dir is not None
        assert label_dir is not None
        _, num_dims = keypoint_helpers.validate_kpt_shape(self.kpt_shape)
        return YOLOKeypointDetectionDatasetArgs(
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            keypoint_set=self.keypoint_set,
            num_dims=num_dims,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
            min_keypoints=self.min_keypoints,
        )

    def get_train_args(self) -> YOLOKeypointDetectionDatasetArgs:
        """Returns dataset args for the training split."""
        return self._get_args(mode="train")

    def get_val_args(self) -> YOLOKeypointDetectionDatasetArgs:
        """Returns dataset args for the validation split."""
        return self._get_args(mode="val")


class YOLOKeypointDetectionDatasetArgs(TaskDatasetArgs):
    """Dataset arguments for a single split of a YOLO keypoint detection dataset."""

    image_dir: Path
    label_dir: Path
    classes: dict[int, str]
    keypoint_set: KeypointSetArgs
    num_dims: int
    ignore_classes: set[int] | None
    skip_if_label_file_missing: bool
    min_keypoints: int

    def list_image_info(self) -> Iterable[dict[str, str]]:
        """Yields image info dicts for each image in the image directory.

        - Bboxes are (x_center, y_center, width, height), keypoints are (x, y).
        - Both are already normalized by the YOLO format and pass through unchanged.
        - Keypoints are not clipped to [0, 1]: one annotated outside the frame keeps
          its position, so transforms can decide what to do with it.
        - Keypoints with visibility 0 sit at (0, 0) and must not be read.
        """
        class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.classes.keys(),
                ignore_classes=self.ignore_classes,
            )
        )
        num_keypoints = self.keypoint_set.num_keypoints

        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / image_filename
            label_filepath = (self.label_dir / image_filename).with_suffix(".txt")

            if label_filepath.exists():
                bboxes, keypoints, visibility, class_labels = (
                    file_helpers.open_yolo_keypoint_detection_label(
                        label_path=label_filepath,
                        num_keypoints=num_keypoints,
                        num_dims=self.num_dims,
                    )
                )
            else:
                # TODO (Thomas, 10/25): Log warning if label file does not exist.
                #   And keep track of how many files are missing labels.
                if self.skip_if_label_file_missing:
                    continue
                bboxes, keypoints, visibility, class_labels = [], [], [], []

            # Drop instances of excluded classes and instances with too few labeled
            # keypoints.
            keep = [
                label in class_id_to_internal_class_id
                and _count_labeled(instance_visibility) >= self.min_keypoints
                for label, instance_visibility in zip(class_labels, visibility)
            ]
            bboxes, class_labels, keypoints, visibility = _filter_instances(
                keep=keep,
                bboxes=bboxes,
                class_labels=class_labels,
                keypoints=keypoints,
                visibility=visibility,
            )

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield _image_info(
                image_path=image_filepath,
                bboxes=bboxes,
                class_labels=class_labels,
                keypoints=keypoints,
                visibility=visibility,
                num_keypoints=num_keypoints,
            )

    @staticmethod
    def get_dataset_cls() -> type[KeypointDetectionDataset]:
        return KeypointDetectionDataset


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

    def _get_args(self, split: COCOSplitArgs) -> COCOKeypointDetectionDatasetArgs:
        return COCOKeypointDetectionDatasetArgs(
            labels=Path(split.annotations),
            data_dir=Path(split.images) if split.images is not None else None,
            classes=self._classes,
            keypoint_set=self.keypoint_set,
            ignore_classes=self.ignore_classes,
            skip_if_annotations_missing=self.skip_if_annotations_missing,
            include_crowd=self.include_crowd,
            min_keypoints=self.min_keypoints,
        )

    def get_train_args(self) -> COCOKeypointDetectionDatasetArgs:
        """Returns dataset args for the training split."""
        return self._get_args(split=self.train)

    def get_val_args(self) -> COCOKeypointDetectionDatasetArgs:
        """Returns dataset args for the validation split."""
        return self._get_args(split=self.val)

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


class COCOKeypointDetectionDatasetArgs(TaskDatasetArgs):
    """Dataset arguments for a single split of a COCO keypoint detection dataset."""

    labels: Path
    data_dir: Path | None
    classes: dict[int, str]
    keypoint_set: KeypointSetArgs
    ignore_classes: set[int] | None
    skip_if_annotations_missing: bool
    include_crowd: bool
    min_keypoints: int

    def list_image_info(self) -> Iterable[dict[str, str]]:
        """Yields image info dicts for each image in the COCO annotation file.

        - Coordinates are converted from COCO pixels to normalized [0, 1].
        - Bboxes go from (x, y, width, height) to (x_center, y_center, width, height).
        - Keypoints are not clipped: one annotated outside the frame keeps its
          position, so transforms can decide what to do with it.
        - Keypoints with visibility 0 sit at (0, 0) and must not be read.
        - Instances with a missing or degenerate bbox get one derived from their
          labeled keypoints. Instances with neither are dropped.
        - Images without annotations are kept unless ``skip_if_annotations_missing``.
        """
        class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.classes.keys(),
                ignore_classes=self.ignore_classes,
            )
        )
        num_keypoints = self.keypoint_set.num_keypoints

        with open(self.labels) as f:
            labels_dict = json.load(f)

        annotations_by_image_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        for annotation in labels_dict.get("annotations", []):
            annotations_by_image_id[annotation["image_id"]].append(annotation)

        image_dir = file_helpers.resolve_coco_images_dir(self.labels, self.data_dir)

        for image in labels_dict["images"]:
            image_width_pixel = image["width"]
            image_height_pixel = image["height"]
            image_id = image["id"]
            image_filepath = image_dir / image["file_name"]

            bboxes: list[list[float]] = []
            class_labels: list[int] = []
            keypoints: list[list[list[float]]] = []
            visibility: list[list[int]] = []

            if image_id in annotations_by_image_id:
                for annotation in annotations_by_image_id[image_id]:
                    if annotation.get("iscrowd", 0) and not self.include_crowd:
                        continue

                    instance_keypoints, instance_visibility = self._parse_keypoints(
                        annotation=annotation,
                        num_keypoints=num_keypoints,
                        image_width_pixel=image_width_pixel,
                        image_height_pixel=image_height_pixel,
                    )
                    if _count_labeled(instance_visibility) < self.min_keypoints:
                        continue

                    bbox = self._parse_bbox(
                        annotation=annotation,
                        keypoints=instance_keypoints,
                        visibility=instance_visibility,
                        image_width_pixel=image_width_pixel,
                        image_height_pixel=image_height_pixel,
                    )
                    if bbox is None:
                        # No usable box and no labeled keypoint, nothing to learn.
                        continue

                    bboxes.append(bbox)
                    class_labels.append(annotation["category_id"])
                    keypoints.append(instance_keypoints)
                    visibility.append(instance_visibility)
            else:
                # TODO (Simon, 04/26): Log warning if annotations do not exist for an
                #   image. And keep track of how many images are missing annotations.
                if self.skip_if_annotations_missing:
                    continue

            # Drop instances of excluded classes.
            keep = [label in class_id_to_internal_class_id for label in class_labels]
            bboxes, class_labels, keypoints, visibility = _filter_instances(
                keep=keep,
                bboxes=bboxes,
                class_labels=class_labels,
                keypoints=keypoints,
                visibility=visibility,
            )

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield _image_info(
                image_path=image_filepath,
                bboxes=bboxes,
                class_labels=class_labels,
                keypoints=keypoints,
                visibility=visibility,
                num_keypoints=num_keypoints,
            )

    def _parse_keypoints(
        self,
        annotation: dict[str, Any],
        num_keypoints: int,
        image_width_pixel: float,
        image_height_pixel: float,
    ) -> tuple[list[list[float]], list[int]]:
        """Returns normalized (x, y) keypoints and visibility flags for one annotation.

        An annotation without a ``keypoints`` field has no labeled keypoints. Common in
        files that mix keypoint and non-keypoint categories, so not an error.
        """
        flat = annotation.get("keypoints")
        if not flat:
            return (
                [[0.0, 0.0] for _ in range(num_keypoints)],
                [keypoint_helpers.VISIBILITY_UNLABELED] * num_keypoints,
            )

        if len(flat) != 3 * num_keypoints:
            raise ValueError(
                f"Expected 'keypoints' of annotation {annotation.get('id')} in "
                f"'{self.labels}' to have {3 * num_keypoints} values for "
                f"{num_keypoints} keypoints, got {len(flat)}."
            )

        keypoints: list[list[float]] = []
        visibility: list[int] = []
        for i in range(num_keypoints):
            x, y, vis = flat[3 * i], flat[3 * i + 1], int(flat[3 * i + 2])
            if vis not in keypoint_helpers.VISIBILITIES:
                raise ValueError(
                    f"Expected keypoint visibility to be one of "
                    f"{list(keypoint_helpers.VISIBILITIES)}, got {vis} for keypoint "
                    f"{i} of annotation {annotation.get('id')} in '{self.labels}'."
                )
            if vis == keypoint_helpers.VISIBILITY_UNLABELED:
                keypoints.append([0.0, 0.0])
            else:
                keypoints.append([x / image_width_pixel, y / image_height_pixel])
            visibility.append(vis)

        # COCO stores the labeled keypoint count, but third-party exports are not
        # always consistent about it, so the visibility flags win.
        num_labeled = _count_labeled(visibility)
        stored = annotation.get("num_keypoints")
        if stored is not None and stored != num_labeled:
            logger.debug(
                f"Annotation {annotation.get('id')} in '{self.labels}' declares "
                f"num_keypoints={stored} but has {num_labeled} keypoints with a "
                f"non-zero visibility flag. Using {num_labeled}."
            )
        return keypoints, visibility

    def _parse_bbox(
        self,
        annotation: dict[str, Any],
        keypoints: list[list[float]],
        visibility: list[int],
        image_width_pixel: float,
        image_height_pixel: float,
    ) -> list[float] | None:
        """Returns the normalized (x_center, y_center, w, h) box for one annotation.

        Falls back to the tight box around the labeled keypoints when the box is
        missing or degenerate. Returns None if neither is available.
        """
        bbox = annotation.get("bbox")
        if bbox is not None:
            left_pixel, top_pixel, width_pixel, height_pixel = bbox
            if width_pixel > 0 and height_pixel > 0:
                return [
                    (left_pixel + width_pixel / 2.0) / image_width_pixel,
                    (top_pixel + height_pixel / 2.0) / image_height_pixel,
                    width_pixel / image_width_pixel,
                    height_pixel / image_height_pixel,
                ]
        return keypoint_helpers.bbox_from_keypoints(
            keypoints_xy=keypoints, visibility=visibility
        )

    @staticmethod
    def get_dataset_cls() -> type[KeypointDetectionDataset]:
        return KeypointDetectionDataset
