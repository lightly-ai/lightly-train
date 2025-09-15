#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from albumentations import BboxParams, Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
    YOLOObjectDetectionDataset,
)
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
    ObjectDetectionTransformInput,
    ObjectDetectionTransformOutput,
)
from lightly_train._transforms.transform import (
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ResizeArgs,
)

from ..helpers import create_yolo_dataset


class DummyTransform(ObjectDetectionTransform):
    transform_args_cls = ObjectDetectionTransformArgs

    def __init__(self, transform_args: ObjectDetectionTransformArgs):
        super().__init__(transform_args=transform_args)
        self.transform = Compose(
            [
                Resize(32, 32),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ],
            bbox_params=transform_args.bbox_params,
        )

    def __call__(
        self, input: ObjectDetectionTransformInput
    ) -> ObjectDetectionTransformOutput:
        output: ObjectDetectionTransformOutput = self.transform(**input)
        return output


class DummyTransformArgs(ObjectDetectionTransformArgs):
    photometric_distort: RandomPhotometricDistortArgs | None = None
    random_zoom_out: RandomZoomOutArgs | None = None
    random_flip: RandomFlipArgs | None = None
    resize: ResizeArgs | None = None
    bbox_params: BboxParams = BboxParams(
        format="yolo",
        label_fields=["class_labels"],
    )


class TestYoloObjectDetectionDataset:
    def test__split_first(self, tmp_path: Path) -> None:
        create_yolo_dataset(tmp_path=tmp_path, split_first=True)

        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train="train/images",
            val="val/images",
            names={0: "class_0", 1: "class_1"},
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOObjectDetectionDataset(
            dataset_args=train_args,
            transform=DummyTransform(DummyTransformArgs()),
            image_filenames=["0.png", "1.png"],
        )

        val_dataset = YOLOObjectDetectionDataset(
            dataset_args=val_args,
            transform=DummyTransform(DummyTransformArgs()),
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

    def test__split_last(self, tmp_path: Path) -> None:
        create_yolo_dataset(tmp_path=tmp_path, split_first=False)

        args = YOLOObjectDetectionDataArgs(
            path=tmp_path,
            train="images/train",
            val="images/val",
            names={0: "class_0", 1: "class_1"},
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = YOLOObjectDetectionDataset(
            dataset_args=train_args,
            transform=DummyTransform(DummyTransformArgs()),
            image_filenames=["0.png", "1.png"],
        )

        val_dataset = YOLOObjectDetectionDataset(
            dataset_args=val_args,
            transform=DummyTransform(DummyTransformArgs()),
            image_filenames=["0.png", "1.png"],
        )

        sample = train_dataset[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["bboxes"].shape == (1, 4)
        assert sample["classes"].shape == (1,)

        sample = val_dataset[0]
        assert sample["image"].shape == (3, 32, 32)
        assert sample["bboxes"].shape == (1, 4)
