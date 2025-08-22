#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import albumentations as A

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
    YOLOObjectDetectionDataset,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)

from ..helpers import create_yolo_dataset


class DummyTransform(TaskTransform):
    def __init__(self, transform_args: TaskTransformArgs):
        super().__init__(transform_args=transform_args)
        self.transform = A.Compose(
            [
                A.Resize(32, 32),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.pytorch.transforms.ToTensorV2(),
            ],
            bbox_params=self.transform_args.bbox_params,
        )

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        output: TaskTransformOutput = self.transform(**input)
        return output


class TestYoloObjectDetectionDataset:
    def test__split_first(self, tmp_path: Path) -> None:
        create_yolo_dataset(tmp_path=tmp_path, split_first=True)

        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train=Path("train/images"),
            val=Path("val/images"),
            names={0: "class_0", 1: "class_1"},
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOObjectDetectionDataset(
            dataset_args=train_args,
            transform=DummyTransform(TaskTransformArgs()),
            image_filenames=["0.png", "1.png"],
        )

        val_dataset = YOLOObjectDetectionDataset(
            dataset_args=val_args,
            transform=DummyTransform(TaskTransformArgs()),
            image_filenames=["0.png", "1.png"],
        )

        sample = train_dataset[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].shape == (1,)

        sample = val_dataset[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].shape == (1,)
