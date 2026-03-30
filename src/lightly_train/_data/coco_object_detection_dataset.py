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
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field

from lightly_train._data import label_helpers
from lightly_train._data.object_detection_dataset import ObjectDetectionDataset
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDatasetArgs


class COCOObjectDetectionDataArgs(TaskDataArgs):
    """Data arguments for a COCO-format object detection dataset.

    The labels files are COCO JSON annotation files. Images are resolved relative
    to the annotation file's parent directory, optionally under ``train_data_dir``
    or ``val_data_dir``.
    """

    # TODO: (Lionel, 08/25): Handle test set.
    format: Literal["coco"] = "coco"
    train_labels: Path
    train_data_dir: Path | None = None
    val_labels: Path
    val_data_dir: Path | None = None
    ignore_classes: set[int] | None = Field(default=None, strict=False)
    skip_if_annotations_missing: bool = False

    @functools.cached_property
    def _classes(self) -> dict[int, str]:
        """Reads and caches the class mapping from the train labels file.

        Always uses the training labels so that train and validation share the same
        class-to-internal-id mapping.
        """
        with open(self.train_labels) as f:
            return {c["id"]: c["name"] for c in json.load(f).get("categories", [])}

    def train_imgs_path(self) -> Path:
        # TODO (simon 03/26): We currently only need this to calculate a hash for the mmmap file for the dataset.
        #  This might not be the best idea as the contents of the file might change.
        return self.train_labels.resolve()

    def val_imgs_path(self) -> Path:
        # TODO (simon 03/26): We currently only need this to calculate a hash for the mmmap file for the dataset.
        #  This might not be the best idea as the contents of the file might change.
        return self.val_labels.resolve()

    def get_train_args(
        self,
    ) -> COCOObjectDetectionDatasetArgs:
        """Returns dataset args for the training split."""
        return COCOObjectDetectionDatasetArgs(
            labels=self.train_labels,
            data_dir=self.train_data_dir,
            classes=self._classes,
            ignore_classes=self.ignore_classes,
            skip_if_annotations_missing=self.skip_if_annotations_missing,
        )

    def get_val_args(self) -> COCOObjectDetectionDatasetArgs:
        """Returns dataset args for the validation split."""
        return COCOObjectDetectionDatasetArgs(
            labels=self.val_labels,
            data_dir=self.val_data_dir,
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


class COCOObjectDetectionDatasetArgs(TaskDatasetArgs):
    """Dataset arguments for a single split of a COCO-format object detection dataset."""

    labels: Path
    data_dir: Path | None
    classes: dict[int, str]
    ignore_classes: set[int] | None
    skip_if_annotations_missing: bool

    def list_image_info(self) -> Iterable[dict[str, str]]:
        """Yields image info dicts for each image in the COCO annotation file.

        Bounding boxes are converted from COCO format (x, y, width, height in pixels)
        to normalized (x_center, y_center, width, height) format. Images with no
        annotations are included unless ``skip_if_annotations_missing`` is True.
        """
        class_id_to_internal_class_id = (
            label_helpers.get_class_id_to_internal_class_id_mapping(
                class_ids=self.classes.keys(),
                ignore_classes=self.ignore_classes,
            )
        )

        with open(self.labels) as f:
            labels_dict = json.load(f)

        annotations_dict = labels_dict.get("annotations", [])

        annotations_by_image_id = defaultdict(list)
        for annotation in annotations_dict:
            annotations_by_image_id[annotation["image_id"]].append(annotation)

        image_dir = self.labels.resolve().parent
        if self.data_dir is not None:
            image_dir /= self.data_dir

        for image in labels_dict["images"]:
            image_width_pixel = image["width"]
            image_height_pixel = image["height"]
            file_name = image["file_name"]
            image_id = image["id"]

            image_filepath = image_dir / file_name

            bboxes = []
            class_labels = []
            if image_id in annotations_by_image_id:
                for annotation in annotations_by_image_id[image_id]:
                    left_pixel, top_pixel, width_pixel, height_pixel = annotation[
                        "bbox"
                    ]
                    # Convert to (x_center, y_center, width, height) format and normalize by image size.
                    x_center = (left_pixel + width_pixel / 2.0) / image_width_pixel
                    y_center = (top_pixel + height_pixel / 2.0) / image_height_pixel
                    width = width_pixel / image_width_pixel
                    height = height_pixel / image_height_pixel

                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(annotation["category_id"])

            else:
                # TODO (Simon, 03/26): Log warning if annotations do not exist for an image.
                #   And keep track of how many files are missing labels.
                if self.skip_if_annotations_missing:
                    continue

            # Remove instances with class IDs that are not in the included classes
            keep = [label in class_id_to_internal_class_id for label in class_labels]
            bboxes = list(itertools.compress(bboxes, keep))
            class_labels = list(itertools.compress(class_labels, keep))

            # Map class IDs to internal class IDs.
            class_labels = [
                class_id_to_internal_class_id[label] for label in class_labels
            ]

            yield {
                "image_path": str(image_filepath),
                "bboxes": json.dumps(bboxes),
                "class_labels": json.dumps(class_labels),
            }

    @staticmethod
    def get_dataset_cls() -> type[ObjectDetectionDataset]:
        return ObjectDetectionDataset
