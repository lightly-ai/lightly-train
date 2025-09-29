#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import pytest
from albumentations import BboxParams, Compose

from lightly_train._transforms.random_iou_crop import RandomIoUCrop


@pytest.fixture
def bbox_params():
    return BboxParams(format="pascal_voc", label_fields=["classes"])


class TestRandomIoUCrop:
    def test__iou_bigger_than_one(self, bbox_params):
        transform = Compose(
            [RandomIoUCrop(sampler_options=[1.0])], bbox_params=bbox_params
        )
        # Use (height, width, channels) for albumentations
        image = np.random.randn(32, 32, 3)
        boxes = np.array([[10, 10, 20, 20], [5, 5, 15, 15]], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_image = transformed["image"]
        transformed_boxes = np.array(transformed["bboxes"])
        transformed_classes = np.array(transformed["classes"])

        # With min IoU of >= 1.0, no cropping should happen.
        assert np.array_equal(image, transformed_image)
        assert np.array_equal(boxes, transformed_boxes)
        assert np.array_equal(classes, transformed_classes)

    def test_output_types_and_shapes(self, bbox_params):
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.0])], bbox_params=bbox_params
        )
        image = np.random.randn(64, 64, 3).astype(np.float32)
        boxes = np.array([[10, 10, 30, 30], [20, 20, 40, 40]], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_image = transformed["image"]
        transformed_boxes = np.array(transformed["bboxes"])
        transformed_classes = np.array(transformed["classes"])

        # Check dtypes
        assert transformed_image.dtype == image.dtype
        assert transformed_boxes.dtype == boxes.dtype
        assert transformed_classes.dtype == classes.dtype

        # Check shapes
        assert transformed_image.shape[2] == 3
        assert transformed_boxes.shape[1] == 4
        assert transformed_classes.shape == (transformed_boxes.shape[0],)

    def test_crop_with_min_iou_zero(self, bbox_params):
        # With min IoU 0.0, cropping is allowed, so output may differ from input.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.0])], bbox_params=bbox_params
        )
        image = np.random.randn(32, 32, 3).astype(np.float32)
        boxes = np.array([[5, 5, 25, 25]], dtype=np.float32)
        classes = np.array([1], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_image = transformed["image"]
        transformed_boxes = np.array(transformed["bboxes"])

        # Output image shape should be (h, w, 3)
        assert transformed_image.ndim == 3
        assert transformed_image.shape[2] == 3
        # Output boxes shape should be (N, 4)
        assert transformed_boxes.shape[1] == 4

    def test_crop_with_no_boxes(self, bbox_params):
        # If there are no boxes, output should be unchanged.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.0])], bbox_params=bbox_params
        )
        image = np.random.randn(32, 32, 3).astype(np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        assert np.array_equal(transformed["image"], image)
        assert transformed["bboxes"] == []
        assert np.array_equal(transformed["classes"], classes)

    def test_crop_with_min_iou_one(self, bbox_params):
        # Already covered by test__iou_bigger_than_one, but check types as well.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[1.0])], bbox_params=bbox_params
        )
        image = np.random.randn(16, 16, 3).astype(np.float32)
        boxes = np.array([[2, 2, 10, 10]], dtype=np.float32)
        classes = np.array([1], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        assert np.array_equal(transformed["image"], image)
        assert np.array_equal(np.array(transformed["bboxes"]), boxes)
        assert np.array_equal(np.array(transformed["classes"]), classes)

    def test_crop_does_not_remove_all_boxes(self, bbox_params):
        # The transform should never return zero boxes if there was at least one input box.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.5])], bbox_params=bbox_params
        )
        image = np.random.randn(32, 32, 3).astype(np.float32)
        boxes = np.array([[5, 5, 25, 25], [10, 10, 20, 20]], dtype=np.float32)
        classes = np.array([1, 2], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        assert len(transformed["bboxes"]) > 0
        assert len(transformed["classes"]) == len(transformed["bboxes"])
