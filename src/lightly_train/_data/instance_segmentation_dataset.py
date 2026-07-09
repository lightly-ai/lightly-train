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
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal, Sequence

if sys.version_info >= (3, 9):
    from pycocotools import mask as coco_mask
else:
    coco_mask: Any = None

import numpy as np
import numpy.typing as npt
import pydantic
import torch
from pydantic import Field

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import data_helpers, file_helpers, label_helpers, yolo_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._env import Env
from lightly_train._transforms.eomt_transforms.instance_segmentation import (
    EoMTInstanceSegmentationCollateFunction,
    EoMTInstanceSegmentationTransform,
    EoMTInstanceSegmentationTransformArgs,
    EoMTInstanceSegmentationTransformInput,
    EoMTInstanceSegmentationTransformOutput,
)
from lightly_train._transforms.ltdetr_transforms.instance_segmentation import (
    LTDETRInstanceSegmentationCollateFunction,
    LTDETRInstanceSegmentationTransform,
    LTDETRInstanceSegmentationTransformArgs,
    LTDETRInstanceSegmentationTransformInput,
    LTDETRInstanceSegmentationTransformOutput,
)
from lightly_train._transforms.task_transform import TaskCollateFunction, TaskTransform
from lightly_train.types import (
    BinaryMasksDict,
    InstanceSegmentationDatasetItem,
    PathLike,
)


def _filter_valid_polygon_segments(
    segmentation: list[list[float]],
) -> list[list[float]]:
    """Return polygon segments with >= 3 points (>= 6 even-length coords)."""
    return [
        segment
        for segment in segmentation
        if isinstance(segment, list) and len(segment) >= 6 and len(segment) % 2 == 0
    ]


def _normalize_polygon_segments(
    segments: list[list[float]],
    image_width: float,
    image_height: float,
) -> list[list[float]]:
    """Normalize polygon coordinates from pixels to [0, 1]."""
    return [
        [
            coord / image_width if i % 2 == 0 else coord / image_height
            for i, coord in enumerate(segment)
        ]
        for segment in segments
    ]


def _bbox_from_polygon_segments(
    segments: list[list[float]],
) -> tuple[float, float, float, float]:
    """Compute [x, y, w, h] bounding box in pixels from polygon segments."""
    all_px = [coord for segment in segments for coord in segment]
    xs = all_px[0::2]
    ys = all_px[1::2]
    left = min(xs)
    top = min(ys)
    return left, top, max(xs) - left, max(ys) - top


def _process_polygon_segmentation(
    segmentation: list[list[float]],
    annotation: dict[str, Any],
    image_width: float,
    image_height: float,
) -> tuple[list[list[float]], tuple[float, float, float, float]] | None:
    """Process a polygon segmentation annotation.

    Returns the normalized polygon group and bbox in [x, y, w, h] pixels,
    or None if no valid segments exist.
    """
    valid_segments = _filter_valid_polygon_segments(segmentation)
    if not valid_segments:
        return None
    polygon_group_norm = _normalize_polygon_segments(
        valid_segments, image_width, image_height
    )
    if "bbox" in annotation:
        bbox = tuple(annotation["bbox"])
    else:
        bbox = _bbox_from_polygon_segments(valid_segments)
    return polygon_group_norm, bbox  # type: ignore[return-value]


def _process_rle_segmentation(
    segmentation: dict[str, Any],
    annotation: dict[str, Any],
    image_width: float,
    image_height: float,
) -> tuple[dict[str, Any], tuple[float, float, float, float]]:
    """Process an RLE segmentation annotation.

    Returns the compressed RLE dict and bbox in [x, y, w, h] pixels.
    """
    if coco_mask is None:
        raise RuntimeError(
            "RLE encoded segmentation requires Python >= 3.9 for pycocotools support."
        )
    if isinstance(segmentation.get("counts"), list):
        rle = coco_mask.frPyObjects(  # type: ignore[call-overload]
            segmentation, image_height, image_width
        )
    else:
        rle = segmentation
    if "bbox" in annotation:
        bbox = tuple(annotation["bbox"])
    else:
        bbox = tuple(coco_mask.toBbox(rle).flatten().tolist())
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle, bbox  # type: ignore[return-value]


class InstanceSegmentationDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: (  # type: ignore[assignment]
        COCOInstanceSegmentationDatasetArgs | YOLOInstanceSegmentationDatasetArgs
    )

    def __init__(
        self,
        dataset_args: COCOInstanceSegmentationDatasetArgs
        | YOLOInstanceSegmentationDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: EoMTInstanceSegmentationTransform
        | LTDETRInstanceSegmentationTransform
        | None = None,
    ) -> None:
        super().__init__(
            transform=transform, dataset_args=dataset_args, image_info=image_info
        )
        if transform is not None:
            self._init_image_mode(transform)

    def set_transform(self, transform: TaskTransform) -> None:
        super().set_transform(transform)
        assert isinstance(
            transform,
            (EoMTInstanceSegmentationTransform, LTDETRInstanceSegmentationTransform),
        )
        self._init_image_mode(transform)

    def get_batch_collate_fn_cls(self) -> type[TaskCollateFunction]:
        if isinstance(self.transform, LTDETRInstanceSegmentationTransform):
            return LTDETRInstanceSegmentationCollateFunction
        return EoMTInstanceSegmentationCollateFunction

    def _init_image_mode(
        self,
        transform: EoMTInstanceSegmentationTransform
        | LTDETRInstanceSegmentationTransform,
    ) -> None:
        transform_args = transform.transform_args
        assert isinstance(
            transform_args,
            (
                EoMTInstanceSegmentationTransformArgs,
                LTDETRInstanceSegmentationTransformArgs,
            ),
        )

        image_mode = (
            None
            if Env.LIGHTLY_TRAIN_IMAGE_MODE.value is None
            else ImageMode(Env.LIGHTLY_TRAIN_IMAGE_MODE.value)
        )
        if image_mode is None:
            image_mode = (
                ImageMode.RGB
                if transform_args.num_channels == 3
                else ImageMode.UNCHANGED
            )

        if image_mode not in (ImageMode.RGB, ImageMode.UNCHANGED):
            raise ValueError(
                f"Invalid image mode: '{image_mode}'. "
                f"Supported modes are '{[ImageMode.RGB.value, ImageMode.UNCHANGED.value]}'."
            )
        self.image_mode = image_mode

    def __getitem__(self, index: int) -> InstanceSegmentationDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])
        segments = json.loads(image_info["segments"])
        bboxes = json.loads(image_info["bboxes"])
        class_labels = json.loads(image_info["class_labels"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

        image_np = file_helpers.open_image_numpy(
            image_path=image_path, mode=self.image_mode
        )

        bboxes_np = np.array(bboxes, dtype=np.float64).reshape(len(bboxes), 4)
        class_labels_np = np.array(class_labels, dtype=np.int64)

        h, w = image_np.shape[0], image_np.shape[1]
        mask_list: list[npt.NDArray[np.bool_]] = []
        for segment in segments:
            if isinstance(segment, list):
                # Polygon format: list of polygon coordinate arrays.
                polygons_np = [np.array(poly, dtype=np.float64) for poly in segment]
                mask = yolo_helpers.binary_mask_from_polygon(
                    polygons_np, height=h, width=w
                )
            elif isinstance(segment, dict):
                # Compressed RLE format.
                if coco_mask is None:
                    raise RuntimeError(
                        "RLE encoded segmentation requires Python >= 3.9 "
                        "for pycocotools support."
                    )
                mask = coco_mask.decode(segment).astype(np.bool_)  # type: ignore[arg-type]
            mask_list.append(mask)
        binary_masks_np = (
            np.stack(mask_list) if mask_list else np.zeros((0, h, w), dtype=np.bool_)
        )

        transform_input: (
            EoMTInstanceSegmentationTransformInput
            | LTDETRInstanceSegmentationTransformInput
        ) = {
            "image": image_np,
            # Shape (n_instances, H, W)
            "binary_masks": binary_masks_np.astype(np.uint8),
            "bboxes": bboxes_np,  # Shape (n_instances, 4)
            "class_labels": class_labels_np,  # Shape (n_instances,)
        }

        transformed: (
            EoMTInstanceSegmentationTransformOutput
            | LTDETRInstanceSegmentationTransformOutput
        ) = self.transform(transform_input)

        image = transformed["image"]
        # Some albumentations versions return lists of tuples instead of arrays.
        if isinstance(transformed["bboxes"], list):
            transformed["bboxes"] = np.array(transformed["bboxes"])
        if isinstance(transformed["class_labels"], list):
            transformed["class_labels"] = np.array(transformed["class_labels"])

        bboxes = torch.from_numpy(transformed["bboxes"]).float()
        class_labels = torch.from_numpy(transformed["class_labels"]).long()
        # Match format from MaskSemanticSegmentationDatasetItem
        binary_masks: BinaryMasksDict = {
            "masks": transformed["binary_masks"].bool(),
            "labels": class_labels,
        }

        return InstanceSegmentationDatasetItem(
            image_path=str(image_path),
            image=image,
            binary_masks=binary_masks,
            bboxes=bboxes,
            classes=class_labels,
        )


class YOLOInstanceSegmentationDataArgs(TaskDataArgs):
    format: Literal["yolo"] = "yolo"
    ignore_index: ClassVar[int | None] = None
    path: PathLike
    train: PathLike
    val: PathLike
    # TODO(Guarin, 10/25): Handle test set.
    test: PathLike | None = None
    # "names" instead of "classes" to match YOLO convention.
    names: dict[int, str]
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_label_file_missing: bool = False

    def resolve_data_paths(self, base_dir: Path) -> None:
        self.path = data_helpers.resolve_path(self.path, base_dir=base_dir)

    def train_data_mmap_hash(self) -> str:
        return str(
            (
                (Path(self.path) / self.train).resolve(),
                self.names,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_label_file_missing,
            )
        )

    def val_data_mmap_hash(self) -> str:
        return str(
            (
                (Path(self.path) / self.val).resolve(),
                self.names,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_label_file_missing,
            )
        )

    @pydantic.field_validator("train", "val", mode="after")
    def validate_paths(cls, v: PathLike) -> Path:
        v = Path(v)
        if "images" not in v.parts:
            raise ValueError(f"Expected path to include 'images' directory, got {v}.")
        return v

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

    def get_train_args(
        self,
    ) -> YOLOInstanceSegmentationDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="train",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOInstanceSegmentationDatasetArgs(
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
        )

    def get_val_args(self) -> YOLOInstanceSegmentationDatasetArgs:
        image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
            path=Path(self.path),
            train=Path(self.train),
            val=Path(self.val),
            test=Path(self.test) if self.test else None,
            mode="val",
        )
        assert image_dir is not None
        assert label_dir is not None
        return YOLOInstanceSegmentationDatasetArgs(
            image_dir=image_dir,
            label_dir=label_dir,
            classes=self.names,
            ignore_classes=self.ignore_classes,
            skip_if_label_file_missing=self.skip_if_label_file_missing,
        )


class YOLOInstanceSegmentationDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    label_dir: Path
    classes: dict[int, str]
    ignore_classes: set[int] | None
    skip_if_label_file_missing: bool

    def list_image_info(self) -> Iterable[dict[str, str]]:
        class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.classes.keys(),
                ignore_classes=self.ignore_classes,
            )
        )

        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / Path(image_filename)
            label_filepath = self.label_dir / Path(image_filename).with_suffix(".txt")

            if label_filepath.exists():
                segments, bboxes, class_labels = (
                    file_helpers.open_yolo_instance_segmentation_label(
                        label_path=label_filepath
                    )
                )
            else:
                # TODO (Thomas, 10/25): Log warning if label file does not exist.
                #   And keep track of how many files are missing labels.
                if self.skip_if_label_file_missing:
                    continue
                segments = []
                bboxes = []
                class_labels = []

            # Remove instances with class IDs that are not in the included classes
            keep = [label in class_id_to_internal_class_id for label in class_labels]
            segments = list(itertools.compress(segments, keep))
            bboxes = list(itertools.compress(bboxes, keep))
            class_labels = list(itertools.compress(class_labels, keep))

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield {
                "image_path": str(image_filepath),
                "segments": json.dumps(segments),
                "bboxes": json.dumps(bboxes),
                "class_labels": json.dumps(class_labels),
            }

    @staticmethod
    def get_dataset_cls() -> type[InstanceSegmentationDataset]:
        return InstanceSegmentationDataset


class COCOSplitArgs(PydanticConfig):
    annotations: PathLike
    images: PathLike | None = None


class COCOInstanceSegmentationDataArgs(TaskDataArgs):
    """Data arguments for a COCO-format instance segmentation dataset.

    The labels files are COCO JSON annotation files. Images are resolved relative
    to the annotation file's parent directory, optionally under ``images``.
    """

    format: Literal["coco"] = "coco"
    train: COCOSplitArgs
    val: COCOSplitArgs
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_annotations_missing: bool = False

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
    def _classes(self) -> dict[int, str]:
        """Reads and caches the class mapping from the train labels file.

        Always uses the training labels so that train and validation share the same
        class-to-internal-id mapping.
        """
        with open(self.train.annotations) as f:
            return {c["id"]: c["name"] for c in json.load(f).get("categories", [])}

    def train_data_mmap_hash(self) -> str:
        annotations_path = Path(self.train.annotations).resolve()
        images_dir = file_helpers.resolve_coco_images_dir(
            annotations_path, self.train.images
        )
        return str(
            (
                annotations_path,
                annotations_path.stat().st_mtime,
                images_dir,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_annotations_missing,
            )
        )

    def val_data_mmap_hash(self) -> str:
        annotations_path = Path(self.val.annotations).resolve()
        images_dir = file_helpers.resolve_coco_images_dir(
            annotations_path, self.val.images
        )
        return str(
            (
                annotations_path,
                annotations_path.stat().st_mtime,
                images_dir,
                sorted(self.ignore_classes) if self.ignore_classes else None,
                self.skip_if_annotations_missing,
            )
        )

    def get_train_args(self) -> COCOInstanceSegmentationDatasetArgs:
        """Returns dataset args for the training split."""
        return COCOInstanceSegmentationDatasetArgs(
            labels=Path(self.train.annotations),
            data_dir=Path(self.train.images) if self.train.images is not None else None,
            classes=self._classes,
            ignore_classes=self.ignore_classes,
            skip_if_annotations_missing=self.skip_if_annotations_missing,
        )

    def get_val_args(self) -> COCOInstanceSegmentationDatasetArgs:
        """Returns dataset args for the validation split."""
        return COCOInstanceSegmentationDatasetArgs(
            labels=Path(self.val.annotations),
            data_dir=Path(self.val.images) if self.val.images is not None else None,
            classes=self._classes,
            ignore_classes=self.ignore_classes,
            skip_if_annotations_missing=self.skip_if_annotations_missing,
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


class COCOInstanceSegmentationDatasetArgs(TaskDatasetArgs):
    """Dataset arguments for a single split of a COCO-format instance segmentation dataset."""

    labels: Path
    data_dir: Path | None
    classes: dict[int, str]
    ignore_classes: set[int] | None
    skip_if_annotations_missing: bool

    def list_image_info(self) -> Iterable[dict[str, str]]:
        """Yields image info dicts for each image in the COCO annotation file.

        Each annotation's segmentation is either a polygon group or an RLE dict.
        Polygon segmentations are normalized from pixel coordinates to [0, 1] and
        stored as ``list[list[float]]``. RLE segmentations are stored as compressed
        RLE dicts (``{"counts": str, "size": [h, w]}``). Bounding boxes are
        converted from COCO format (x, y, width, height in pixels) to normalized
        (x_center, y_center, width, height) format. Images with no annotations are
        included unless ``skip_if_annotations_missing`` is True.
        """
        class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.classes.keys(),
                ignore_classes=self.ignore_classes,
            )
        )

        with open(self.labels) as f:
            labels_dict = json.load(f)

        annotations_by_image_id: defaultdict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        for annotation in labels_dict.get("annotations", []):
            annotations_by_image_id[annotation["image_id"]].append(annotation)

        image_dir = self.labels.resolve().parent
        if self.data_dir is not None:
            image_dir /= self.data_dir

        for image in labels_dict["images"]:
            image_width_pixel = image["width"]
            image_height_pixel = image["height"]
            image_id = image["id"]
            image_filepath = image_dir / image["file_name"]

            segments: list[list[list[float]] | dict[str, Any]] = []
            bboxes = []
            class_labels = []
            if image_id in annotations_by_image_id:
                for annotation in annotations_by_image_id[image_id]:
                    segmentation = annotation.get("segmentation", [])
                    if not segmentation:
                        continue
                    if isinstance(segmentation, list):
                        result = _process_polygon_segmentation(
                            segmentation=segmentation,
                            annotation=annotation,
                            image_width=image_width_pixel,
                            image_height=image_height_pixel,
                        )
                        if result is None:
                            continue
                        segment: list[list[float]] | dict[str, Any] = result[0]
                        bbox = result[1]
                    elif isinstance(segmentation, dict):
                        segment, bbox = _process_rle_segmentation(
                            segmentation=segmentation,
                            annotation=annotation,
                            image_width=image_width_pixel,
                            image_height=image_height_pixel,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported segmentation format: {type(segmentation)}. "
                            "Expected a list of polygons or an RLE dict."
                        )

                    segments.append(segment)

                    # Convert bbox from [x, y, w, h] pixels to normalized
                    # [x_center, y_center, w, h].
                    left_pixel, top_pixel, width_pixel, height_pixel = bbox
                    x_center = (left_pixel + width_pixel / 2.0) / image_width_pixel
                    y_center = (top_pixel + height_pixel / 2.0) / image_height_pixel
                    width = width_pixel / image_width_pixel
                    height = height_pixel / image_height_pixel
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(annotation["category_id"])
            else:
                # TODO (Simon, 04/26): Log warning if annotations do not exist for an image.
                #   And keep track of how many images are missing annotations.
                if self.skip_if_annotations_missing:
                    continue

            # Remove instances with class IDs that are not in the included classes.
            keep = [label in class_id_to_internal_class_id for label in class_labels]
            segments = list(itertools.compress(segments, keep))
            bboxes = list(itertools.compress(bboxes, keep))
            class_labels = list(itertools.compress(class_labels, keep))

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield {
                "image_path": str(image_filepath),
                "segments": json.dumps(segments),
                "bboxes": json.dumps(bboxes),
                "class_labels": json.dumps(class_labels),
            }

    @staticmethod
    def get_dataset_cls() -> type[InstanceSegmentationDataset]:
        return InstanceSegmentationDataset
