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
def bbox_params() -> BboxParams:
    return BboxParams(format="pascal_voc", label_fields=["classes"])


class TestRandomIoUCrop:
    def test__iou_bigger_than_one(self, bbox_params: BboxParams) -> None:
        transform = Compose(
            [RandomIoUCrop(sampler_options=[1.0])], bbox_params=bbox_params
        )
        # Use (height, width, channels) for albumentations
        image = np.random.randn(32, 32, 3)
        boxes = np.array([[10, 10, 20, 20], [5, 5, 15, 15]], dtype=np.float64)
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

    def test_output_types_and_shapes(self, bbox_params: BboxParams) -> None:
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.0])], bbox_params=bbox_params
        )
        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        boxes = np.array([[10, 10, 30, 30], [20, 20, 40, 40]], dtype=np.float64)
        classes = np.array([1, 2], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_image = transformed["image"]
        transformed_boxes = transformed["bboxes"]
        transformed_classes = transformed["classes"]

        # Convert to arrays if needed (for compatibility with older albumentations)
        transformed_boxes_arr = np.array(transformed_boxes)
        transformed_classes_arr = np.array(transformed_classes)

        # Check dtypes
        assert transformed_image.dtype == image.dtype
        # Some albumentations versions return float32, others float64.
        assert transformed_boxes_arr.dtype in {np.dtype("float32"), np.dtype("float64")}
        assert np.all(
            np.equal(np.mod(transformed_classes_arr, 1), 0)
        )  # check if integers

        # Check shapes
        assert transformed_image.shape[2] == 3
        assert transformed_boxes_arr.shape[1] == 4
        assert transformed_classes_arr.shape == (transformed_boxes_arr.shape[0],)

    def test_crop_with_min_iou_zero(self, bbox_params: BboxParams) -> None:
        # With min IoU 0.0, cropping is allowed, so output may differ from input.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.0])], bbox_params=bbox_params
        )
        image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        boxes = np.array([[5, 5, 25, 25]], dtype=np.float64)
        classes = np.array([1], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_image = transformed["image"]
        transformed_boxes = transformed["bboxes"]

        transformed_boxes_arr = np.array(transformed_boxes)

        # Output image shape should be (h, w, 3)
        assert transformed_image.ndim == 3
        assert transformed_image.shape[2] == 3
        # Output boxes shape should be (N, 4)
        assert transformed_boxes_arr.shape[1] == 4

    def test_crop_with_no_boxes(self, bbox_params: BboxParams) -> None:
        # If there are no boxes, output should be unchanged.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.0])], bbox_params=bbox_params
        )
        image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        boxes = np.zeros((0, 4), dtype=np.float64)
        classes = np.zeros((0,), dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_boxes = transformed["bboxes"]
        transformed_classes = transformed["classes"]

        # Convert to arrays for comparison
        transformed_boxes_arr = np.array(transformed_boxes)
        transformed_classes_arr = np.array(transformed_classes)

        assert np.array_equal(transformed["image"], image)
        assert len(transformed_boxes_arr) == 0
        assert np.array_equal(transformed_classes_arr, classes)

    def test_crop_with_min_iou_one(self, bbox_params: BboxParams) -> None:
        # Already covered by test__iou_bigger_than_one, but check types as well.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[1.0])], bbox_params=bbox_params
        )
        image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        boxes = np.array([[2, 2, 10, 10]], dtype=np.float64)
        classes = np.array([1], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_boxes = transformed["bboxes"]
        transformed_classes = transformed["classes"]

        transformed_boxes_arr = np.array(transformed_boxes)
        transformed_classes_arr = np.array(transformed_classes)

        assert np.array_equal(transformed["image"], image)
        assert np.array_equal(transformed_boxes_arr, boxes)
        assert np.array_equal(transformed_classes_arr, classes)

    def test_crop_does_not_remove_all_boxes(self, bbox_params: BboxParams) -> None:
        # The transform should never return zero boxes if there was at least one input box.
        transform = Compose(
            [RandomIoUCrop(sampler_options=[0.5])], bbox_params=bbox_params
        )
        image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        boxes = np.array([[5, 5, 25, 25], [10, 10, 20, 20]], dtype=np.float64)
        classes = np.array([1, 2], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_boxes = transformed["bboxes"]
        transformed_classes = transformed["classes"]

        # Convert to arrays for length checks
        transformed_boxes_arr = np.array(transformed_boxes)
        transformed_classes_arr = np.array(transformed_classes)

        assert transformed_boxes_arr.shape[0] > 0
        assert transformed_classes_arr.shape[0] == transformed_boxes_arr.shape[0]

    @pytest.mark.skip(
        reason="Disabled as the current test can fail even if the implementation is correct."
    )
    def test_crop_with_p_one_and_forced_scale_change(
        self, bbox_params: BboxParams
    ) -> None:
        """Test that with p=1.0 and scale range that excludes 1.0, transformation always occurs."""
        # Use scale range that doesn't include 1.0 to force cropping.
        # Use larger crop_trials and iou_trials to increase chances of finding a valid crop.
        transform = Compose(
            [
                RandomIoUCrop(
                    min_scale=0.5,
                    max_scale=0.9,
                    sampler_options=[0.0],
                    p=1.0,
                    crop_trials=100,
                    iou_trials=50,
                )
            ],
            bbox_params=bbox_params,
        )
        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        classes = np.array([1], dtype=np.int64)

        # Try multiple times since the crop finding is probabilistic.
        transformation_occurred = False
        for _ in range(10):  # Try up to 10 times.
            data = {"image": image, "bboxes": boxes, "classes": classes}
            transformed = transform(**data)
            transformed_image = transformed["image"]
            transformed_boxes = transformed["bboxes"]
            transformed_classes = transformed["classes"]

            transformed_boxes_arr = np.array(transformed_boxes)
            transformed_classes_arr = np.array(transformed_classes)

            # Check if transformation occurred.
            if not np.array_equal(transformed_image, image) and (
                transformed_image.shape[0] < image.shape[0]
                or transformed_image.shape[1] < image.shape[1]
            ):
                transformation_occurred = True
                # Boxes should be adjusted due to cropping.
                assert not np.array_equal(transformed_boxes_arr, boxes)
                # Number of boxes and classes should remain the same.
                assert transformed_boxes_arr.shape[0] == boxes.shape[0]
                assert transformed_classes_arr.shape[0] == classes.shape[0]
                break

        assert transformation_occurred, (
            "No transformation occurred despite p=1.0 and forced scale change"
        )

    def test_crop_with_p_zero_never_transforms(self, bbox_params: BboxParams) -> None:
        """Test that with p=0.0, no transformation ever occurs."""
        transform = Compose(
            [RandomIoUCrop(min_scale=0.3, max_scale=0.8, sampler_options=[0.0], p=0.0)],
            bbox_params=bbox_params,
        )
        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        boxes = np.array([[10, 10, 50, 50], [20, 20, 40, 40]], dtype=np.float64)
        classes = np.array([1, 2], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)
        transformed_image = transformed["image"]
        transformed_boxes = transformed["bboxes"]
        transformed_classes = transformed["classes"]

        transformed_boxes_arr = np.array(transformed_boxes)
        transformed_classes_arr = np.array(transformed_classes)

        # With p=0.0, nothing should change.
        assert np.array_equal(transformed_image, image)
        assert np.array_equal(transformed_boxes_arr, boxes)
        assert np.array_equal(transformed_classes_arr, classes)

    def test_crop_identity_on_non_square_image(self, bbox_params: BboxParams) -> None:
        """For non-square images and min_scale=max_scale=1.0, the crop should be identity.
        If h/w are swapped anywhere, this will either crash or change output shape.
        """
        transform = Compose(
            [
                RandomIoUCrop(
                    min_scale=1.0,
                    max_scale=1.0,
                    sampler_options=[0.0],
                    p=1.0,
                )
            ],
            bbox_params=bbox_params,
        )

        # Non-square image: height != width
        h, w = 64, 128
        image = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        boxes = np.array([[10, 10, 50, 40]], dtype=np.float64)
        classes = np.array([1], dtype=np.int64)

        data = {"image": image, "bboxes": boxes, "classes": classes}
        transformed = transform(**data)

        transformed_image = transformed["image"]
        transformed_boxes = np.array(transformed["bboxes"])
        transformed_classes = np.array(transformed["classes"])

        # ✅ Expect exact identity, because crop covers full image
        assert np.array_equal(transformed_image, image), "Image changed unexpectedly"
        assert np.array_equal(transformed_boxes, boxes), "Boxes changed unexpectedly"
        assert np.array_equal(transformed_classes, classes)
        assert transformed_image.shape == (h, w, 3), (
            "Shape mismatch (h/w may be swapped)"
        )
