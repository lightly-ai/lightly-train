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
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from lightly_train._data.coco_object_detection_dataset import (
        COCOObjectDetectionDatasetArgs,
    )
    from lightly_train._data.yolo_object_detection_dataset import (
        YOLOObjectDetectionDatasetArgs,
    )

import numpy as np

from lightly_train._data import file_helpers
from lightly_train._data.task_dataset import TaskDataset
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionCollateFunction,
)
from lightly_train._transforms.task_transform import TaskCollateFunction
from lightly_train.types import ObjectDetectionDatasetItem


class ObjectDetectionDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: COCOObjectDetectionDatasetArgs | YOLOObjectDetectionDatasetArgs  # type: ignore[assignment]

    batch_collate_fn_cls: ClassVar[type[TaskCollateFunction]] = (
        ObjectDetectionCollateFunction
    )

    def __getitem__(self, index: int) -> ObjectDetectionDatasetItem:
        # Load the image.
        image_info = self.image_info[index]
        image_path = Path(image_info["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        image_np = file_helpers.open_image_numpy(image_path)
        h, w, _ = image_np.shape

        bboxes = json.loads(image_info["bboxes"])
        class_labels = json.loads(image_info["class_labels"])
        # TODO (simon, 03/26) do we need this assert?
        assert len(bboxes) == len(class_labels)

        bboxes_np = np.array(bboxes, dtype=np.float64).reshape(len(bboxes), 4)
        class_labels_np = np.array(class_labels, dtype=np.int64)

        transformed = self.transform(
            {
                "image": image_np,
                "bboxes": bboxes_np,  # Shape (n_boxes, 4)
                "class_labels": class_labels_np,  # Shape (n_boxes,)
            }
        )

        return ObjectDetectionDatasetItem(
            image_path=str(image_path),
            image=transformed["image"],
            bboxes=transformed["bboxes"],
            classes=transformed["class_labels"],
            # TODO (Thomas, 10/25): Switch to (h, w) for consistency.
            original_size=(w, h),
        )
