#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from lightly_train._data.yolo_object_detection_dataset import (
    YoloObjectDetectionDataset,
    YoloObjectDetectionDatasetArgs,
)
from lightly_train._transforms.task_transform import TaskTransform

from ..helpers import create_yolo_dataset


class TestYoloObjectDetectionDataset:
    def test__split_first(self, tmp_path: Path) -> None:
        create_yolo_dataset(tmp_path=tmp_path, split_first=True)
        names = {0: "class_0", 1: "class_1"}
        args = YoloObjectDetectionDatasetArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names=names,
        )
        YoloObjectDetectionDataset(
            dataset_args=args,
            transform=TaskTransform(),
            mode="train",
        )
