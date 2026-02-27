#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import pytest
import torch
from albumentations import BboxParams, Compose
from torchvision import tv_tensors

from lightly_train._transforms.scale_jitter import ScaleJitter, TorchVisionScaleJitter


class TestRandomScaleJitter:
    def test__call__check_return_shapes_larger(self) -> None:
        img_size = (16, 16)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        bboxes = np.array([[4, 4, 12, 12]], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)

        transform = ScaleJitter(
            sizes=None,
            target_size=img_size,
            scale_range=(2.0, 4.0),
            num_scales=3,
            p=1.0,
        )
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])
        out = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            bbox_params=bbox_params,
            class_labels=classes,
        )

        assert all(out["image"].shape[i] > img.shape[i] for i in (0, 1))
        assert all(out["image"].shape[i] > mask.shape[i] for i in (0, 1))
        assert np.array(out["class_labels"]).shape == classes.shape
        assert np.array(out["bboxes"]).shape == bboxes.shape

    def test__call__check_return_shapes_smaller(self) -> None:
        img_size = (16, 16)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        bboxes = np.array([[4, 4, 12, 12]], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)

        transform = ScaleJitter(
            sizes=None,
            target_size=img_size,
            scale_range=(0.2, 0.7),
            num_scales=3,
            p=1.0,
        )
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])
        out = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            bbox_params=bbox_params,
            class_labels=classes,
        )

        assert all(out["image"].shape[i] < img.shape[i] for i in (0, 1))
        assert all(out["image"].shape[i] < mask.shape[i] for i in (0, 1))
        assert np.array(out["class_labels"]).shape == classes.shape
        assert np.array(out["bboxes"]).shape == bboxes.shape

    def test__call__check_return_shapes_in_sizes(self) -> None:
        img_size = (16, 16)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        bboxes = np.array([[4, 4, 12, 12]], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)

        sizes = [(8, 8), (12, 12), (20, 20)]
        transform = ScaleJitter(
            sizes=sizes,
            target_size=None,
            scale_range=None,
            num_scales=None,
            p=1.0,
        )
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])
        out = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            bbox_params=bbox_params,
            class_labels=classes,
        )

        assert out["image"].shape in [(s[0], s[1], 3) for s in sizes]
        assert out["mask"].shape in [s for s in sizes]
        assert np.array(out["class_labels"]).shape == classes.shape
        assert np.array(out["bboxes"]).shape == bboxes.shape

    def test__call__no_transform_when_p0(self) -> None:
        img_size = (8, 8)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        bboxes = np.array([[1, 1, 2, 2]], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)

        transform = ScaleJitter(
            sizes=None,
            target_size=img_size,
            scale_range=(1.0, 2.0),
            num_scales=3,
            p=0.0,
        )
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])
        out = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            bbox_params=bbox_params,
            class_labels=classes,
        )

        assert np.array_equal(out["image"], img)
        assert np.array_equal(out["mask"], mask)
        assert np.array_equal(out["bboxes"], bboxes)
        assert np.array_equal(out["class_labels"], classes)

    def test__call__always_transform_when_p1(self) -> None:
        img_size = (16, 16)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
        bboxes = np.array([[1, 1, 2, 2]], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])

        transform = Compose(
            [
                ScaleJitter(
                    sizes=None,
                    target_size=img_size,
                    scale_range=(2.0, 4.0),
                    num_scales=2,
                    p=1.0,
                )
            ],
            bbox_params=bbox_params,
        )
        out = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            class_labels=classes,
        )
        assert out["image"].shape != img.shape
        assert out["mask"].shape != mask.shape
        assert np.array_equal(out["class_labels"], classes)
        # With scale >=2.0 the bbox has to change in Pascal VOC format.
        assert not np.array_equal(out["bboxes"], bboxes)

    def test__step_seeding__deterministic(self) -> None:
        img_size = (8, 8)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        bboxes = np.array([[1, 1, 2, 2]], dtype=np.float32)
        classes = np.array([1], dtype=np.int32)
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])

        transform = Compose(
            [
                ScaleJitter(
                    sizes=None,
                    target_size=img_size,
                    scale_range=(1.0, 10.0),
                    num_scales=10,
                    p=1.0,
                    step_seeding=True,
                    seed_offset=42,
                )
            ],
            bbox_params=bbox_params,
        )
        # Set step and get deterministic idx
        transform.transforms[0].step = 5
        out1 = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            class_labels=classes,
        )
        out2 = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            class_labels=classes,
        )
        assert np.array_equal(out1["image"], out2["image"])
        assert np.array_equal(out1["mask"], out2["mask"])
        assert np.array_equal(out1["bboxes"], out2["bboxes"])
        assert np.array_equal(out1["class_labels"], out2["class_labels"])

    def test__step_seeding__different_steps(self) -> None:
        img_size = (8, 8)
        img = np.random.randint(0, 255, size=(*img_size, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=img_size, dtype=np.uint8)
        bboxes = np.array([[1, 1, 2, 2]], dtype=np.float64)
        classes = np.array([1], dtype=np.int64)
        bbox_params = BboxParams(format="pascal_voc", label_fields=["class_labels"])

        transform = Compose(
            [
                ScaleJitter(
                    sizes=None,
                    target_size=img_size,
                    scale_range=(1.0, 10.0),
                    num_scales=10,
                    p=1.0,
                    step_seeding=True,
                    seed_offset=42,
                )
            ],
            bbox_params=bbox_params,
        )
        # Set step and get deterministic idx for first transform
        transform.transforms[0].step = 5
        out1 = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            class_labels=classes,
        )
        # Change step and get deterministic idx for second transform
        transform.transforms[0].step = 6
        out2 = transform(
            image=img,
            mask=mask,
            bboxes=bboxes,
            class_labels=classes,
        )
        assert not np.array_equal(out1["image"], out2["image"])
        assert not np.array_equal(out1["mask"], out2["mask"])
        assert not np.array_equal(out1["bboxes"], out2["bboxes"])
        assert np.array_equal(out1["class_labels"], out2["class_labels"])


@pytest.fixture
def dummy_tv_image() -> tv_tensors.Image:
    return tv_tensors.Image(torch.rand(3, 100, 100))


@pytest.fixture
def dummy_tv_obb() -> tv_tensors.BoundingBoxes:
    return tv_tensors.BoundingBoxes(  # type: ignore[call-arg]
        torch.tensor([[50.0, 50.0, 20.0, 10.0, 45.0]]),
        format=tv_tensors.BoundingBoxFormat.CXCYWHR,
        canvas_size=(100, 100),
    )


@pytest.fixture
def dummy_tv_image_small() -> tv_tensors.Image:
    return tv_tensors.Image(torch.rand(3, 50, 50))


@pytest.fixture
def dummy_tv_obb_small() -> tv_tensors.BoundingBoxes:
    return tv_tensors.BoundingBoxes(  # type: ignore[call-arg]
        torch.tensor([[25.0, 25.0, 10.0, 5.0, 30.0]]),
        format=tv_tensors.BoundingBoxFormat.CXCYWHR,
        canvas_size=(50, 50),
    )


class TestTorchVisionScaleJitter:
    def test_single_input(
        self,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image: tv_tensors.Image,
    ) -> None:
        scale_jitter = TorchVisionScaleJitter(
            target_size=(100, 100),
            scale_range=(0.5, 1.5),
            num_scales=5,
        )

        output = scale_jitter(dummy_tv_image, dummy_tv_obb)

        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[0], tv_tensors.Image)
        assert isinstance(output[1], tv_tensors.BoundingBoxes)

    def test_multiple_inputs_same_resize(
        self,
        dummy_tv_image: tv_tensors.Image,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image_small: tv_tensors.Image,
        dummy_tv_obb_small: tv_tensors.BoundingBoxes,
    ) -> None:
        scale_jitter = TorchVisionScaleJitter(
            target_size=(100, 100),
            scale_range=(0.5, 1.5),
            num_scales=5,
        )

        images = [dummy_tv_image, dummy_tv_image_small]
        bboxes = [dummy_tv_obb, dummy_tv_obb_small]

        outputs = scale_jitter(images, bboxes)

        assert isinstance(outputs, tuple)
        transformed_images, transformed_bboxes = outputs
        assert len(transformed_images) == 2
        assert len(transformed_bboxes) == 2

        shape1 = transformed_images[0].shape
        shape2 = transformed_images[1].shape
        assert shape1 == shape2, (
            f"Expected same shape for all images, got {shape1} and {shape2}"
        )

    def test_random_choice_changes_size(
        self,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image: tv_tensors.Image,
    ) -> None:
        scale_jitter = TorchVisionScaleJitter(
            target_size=(100, 100),
            scale_range=(0.5, 1.5),
            num_scales=5,
        )

        shapes = []
        for _ in range(20):
            output = scale_jitter(dummy_tv_image, dummy_tv_obb)
            shapes.append(output[0].shape)

        unique_shapes = set(shapes)
        assert len(unique_shapes) > 1, (
            "Expected different sizes across iterations, got same size"
        )
