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
from lightly_train._data.keypoint_helpers import KeypointSetArgs
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train.types import KeypointDetectionDatasetItem, PathLike

logger = logging.getLogger(__name__)


class KeypointDetectionDataset(TaskDataset):
    """Dataset for keypoint detection.

    Reading the labels is implemented, but turning them into training samples is not:
    there is no keypoint detection model and no keypoint transform yet, so there is
    nothing to feed. ``list_image_info`` on the dataset args is the useful part today.
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

    They are kept in lockstep; dropping one of them here would misalign the dataset in
    a way that only surfaces during training.
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

    Labels are ``.txt`` files next to the images, in a ``labels`` directory mirroring
    the ``images`` directory. Each line holds one instance as
    ``class_id x_center y_center width height`` followed by ``kpt_shape[0]`` keypoints
    of ``kpt_shape[1]`` values each, all normalized to [0, 1].

    Attributes:
        kpt_shape:
            ``[num_keypoints, num_dims]``. ``num_dims`` is 2 for (x, y) or 3 for
            (x, y, visibility). A dataset with ``num_dims == 2`` cannot express an
            unlabeled keypoint, so all its keypoints are read as visible.
        flip_idx:
            Keypoint permutation applied on a horizontal flip.
        kpt_names:
            Keypoint names per class. A single keypoint set is used for all classes, so
            all included classes must declare the same names.
        keypoints:
            Keypoint set properties that the dataset config has no field for, most
            importantly the OKS ``sigmas`` and the ``skeleton``. Values given both here
            and in the fields above must agree.
        min_keypoints:
            Drop instances with fewer than this many labeled keypoints. Defaults to 0,
            which keeps everything the label file contains.
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
        # Resolve once at construction so that a bad keypoint config fails here rather
        # than much later when the labels are read. The result is not stored: assigning
        # to self would re-trigger validation because of validate_assignment=True.
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

        Bounding boxes are (x_center, y_center, width, height) and keypoints are (x, y),
        both already normalized to [0, 1] by the YOLO format and therefore passed
        through unchanged. Keypoints are not clipped to [0, 1]: a keypoint annotated
        outside the frame keeps its position, so that transforms can decide what to do
        with it. Keypoints with visibility 0 have coordinates (0, 0) and must not be
        read.
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

            # Remove instances with class IDs that are not in the included classes, and
            # instances with too few labeled keypoints.
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

    The labels files are COCO JSON annotation files. Images are resolved relative to the
    annotation file's parent directory, optionally under ``images``.

    The keypoint set is read from the ``categories`` entries of the training
    annotations, which declare the keypoint names and the skeleton. The format declares
    neither OKS sigmas nor flip pairs, so those can only be passed via ``keypoints``.

    Attributes:
        keypoints:
            Keypoint set properties that the annotations do not carry, most importantly
            the OKS ``sigmas`` and the ``flip_idx``. Values given both here and in the
            annotations must agree. Required if no category declares keypoints.
        include_crowd:
            Keep annotations with ``iscrowd == 1``. They describe a group of instances
            rather than one, and carry no usable keypoints, so they are dropped by
            default.
        min_keypoints:
            Drop instances with fewer than this many labeled keypoints. Defaults to 0,
            which keeps everything the annotations contain, including the
            ``num_keypoints == 0`` annotations that COCO keypoint files are full of.
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

        Always uses the training labels so that train and validation share the same
        class-to-internal-id mapping and the same keypoint set. Cached so that the
        annotations are read once for both the classes and the keypoint set.
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

        Keypoints and bounding boxes are converted from COCO's pixel coordinates to
        normalized [0, 1] coordinates, bboxes additionally from (x, y, width, height) to
        (x_center, y_center, width, height). Keypoints are not clipped: a keypoint
        annotated outside the frame keeps its position, so that transforms can decide
        what to do with it. Keypoints with visibility 0 have coordinates (0, 0) and must
        not be read.

        Instances whose bounding box is missing or degenerate get a bounding box derived
        from their labeled keypoints. Instances with neither are dropped. Images with no
        annotations are included unless ``skip_if_annotations_missing`` is True.
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
                        # Neither a usable box nor a labeled keypoint; nothing to learn
                        # from this instance.
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

            # Remove instances with class IDs that are not in the included classes.
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

        An annotation without a ``keypoints`` field has no labeled keypoints. Such
        annotations are common in files that mix keypoint and non-keypoint categories,
        so this is not an error.
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

        # COCO stores the number of labeled keypoints, but third-party exports are not
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

        Falls back to the tight box around the labeled keypoints when the annotation has
        no box or a degenerate one. Returns None if neither is available.
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
