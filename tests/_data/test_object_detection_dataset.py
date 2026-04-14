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
from typing import Literal

import numpy as np
from albumentations import BboxParams

from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDatasetArgs,
)
from lightly_train._data.object_detection_dataset import ObjectDetectionDataset
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
)
from lightly_train.types import ImageSizeTuple

from ..helpers import create_images


class DummyTransformArgs(ObjectDetectionTransformArgs):
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = 3
    photometric_distort: RandomPhotometricDistortArgs | None = None
    random_zoom_out: RandomZoomOutArgs | None = None
    random_iou_crop: RandomIoUCropArgs | None = None
    random_flip: RandomFlipArgs | None = None
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    image_size: ImageSizeTuple = (32, 32)
    scale_jitter: ScaleJitterArgs | None = None
    resize: ResizeArgs | None = None
    normalize: NormalizeArgs | Literal["auto"] | None = None
    bbox_params: BboxParams = BboxParams(
        format="yolo",
        label_fields=["class_labels"],
    )


def test_object_detection_dataset(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    create_images(image_dir=image_dir, files=3)
    image_paths = sorted(image_dir.glob("*.png"))

    image_info = [
        {
            "image_path": str(image_paths[0]),
            "bboxes": json.dumps([]),
            "class_labels": json.dumps([]),
        },
        {
            "image_path": str(image_paths[1]),
            "bboxes": json.dumps([[0.5, 0.5, 0.25, 0.25]]),
            "class_labels": json.dumps([0]),
        },
        {
            "image_path": str(image_paths[2]),
            "bboxes": json.dumps([[0.5, 0.5, 0.25, 0.25], [0.3, 0.7, 0.1, 0.2]]),
            "class_labels": json.dumps([0, 1]),
        },
    ]

    dummy_dataset_args = COCOObjectDetectionDatasetArgs(
        labels=tmp_path / "annotations.json",
        data_dir=None,
        classes={7: "cat", 5: "dog", 6: "horse"},
        ignore_classes=None,
        skip_if_annotations_missing=False,
    )

    dataset = ObjectDetectionDataset(
        dataset_args=dummy_dataset_args,
        transform=ObjectDetectionTransform(DummyTransformArgs()),
        image_info=image_info,
    )

    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["image"].dtype == np.float32
    assert len(sample["bboxes"]) == 0
    assert len(sample["classes"]) == 0

    sample = dataset[1]
    assert sample["image"].dtype == np.float32
    np.testing.assert_array_almost_equal(sample["bboxes"], [[0.5, 0.5, 0.25, 0.25]])
    np.testing.assert_array_equal(sample["classes"], [0])

    sample = dataset[2]
    assert sample["image"].dtype == np.float32
    np.testing.assert_array_almost_equal(
        sample["bboxes"], [[0.5, 0.5, 0.25, 0.25], [0.3, 0.7, 0.1, 0.2]]
    )
    np.testing.assert_array_equal(sample["classes"], [0, 1])
