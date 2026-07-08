#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from albumentations import BboxParams
from numpy.typing import NDArray
from torch import Tensor

from lightly_train._task_models.ltdetr_instance_segmentation.transforms import (
    LTDETRInstanceSegmentationMixUpArgs,
    LTDETRInstanceSegmentationTrainTransformArgs,
)
from lightly_train._transforms.ltdetr_transforms.instance_segmentation import (
    LTDETRInstanceSegmentationCollateFunction,
    LTDETRInstanceSegmentationTransform,
    LTDETRInstanceSegmentationTransformArgs,
    LTDETRInstanceSegmentationTransformInput,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    MosaicArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    RandomZoomOutArgs,
    ResizeArgs,
)
from lightly_train.types import InstanceSegmentationDatasetItem

# Only the mask-aware behavior that is unique to instance segmentation is tested here.
# The shared step-scheduling / dataloader-reinitialization machinery lives in the
# ``_LTDETR*`` base classes and is covered once in ``test_base.py``; the pure helpers in
# ``ltdetr_transforms.utils`` are covered in ``test_utils.py``.


def _get_channel_drop_args() -> ChannelDropArgs:
    return ChannelDropArgs(
        num_channels_keep=3,
        weight_drop=(1.0, 1.0, 0.0, 0.0),
    )


def _get_photometric_distort_args() -> RandomPhotometricDistortArgs:
    return RandomPhotometricDistortArgs(
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
        prob=0.5,
    )


def _get_random_zoom_out_args() -> RandomZoomOutArgs:
    return RandomZoomOutArgs(
        prob=1.0,
        fill=0.0,
        side_range=(1.0, 1.5),
    )


def _get_random_iou_crop_args() -> RandomIoUCropArgs:
    return RandomIoUCropArgs(
        min_scale=0.3,
        max_scale=1.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        sampler_options=None,
        crop_trials=40,
        iou_trials=1000,
        prob=1.0,
    )


def _get_random_flip_args() -> RandomFlipArgs:
    return RandomFlipArgs(horizontal_prob=1.0, vertical_prob=1.0)


def _get_random_rotate_90_args() -> RandomRotate90Args:
    return RandomRotate90Args(prob=1.0)


def _get_random_rotate_args() -> RandomRotationArgs:
    return RandomRotationArgs(prob=1.0, degrees=30.0)


def _get_resize_args() -> ResizeArgs:
    return ResizeArgs(height=64, width=64)


def _get_normalize_args() -> NormalizeArgs:
    return NormalizeArgs()


def _get_mosaic_args() -> MosaicArgs:
    return MosaicArgs(
        prob=1.0,
        step_start=0,
        step_stop=100,
        output_size=32,
        max_size=None,
        rotation_range=10.0,
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),
        fill_value=0,
        max_cached_images=50,
        random_pop=True,
    )


def _get_image_size() -> tuple[int, int]:
    return (64, 64)


def _get_bbox_params() -> BboxParams:
    # Instance segmentation threads an ``indices`` label field alongside
    # ``class_labels`` so masks can be re-aligned after the spatial transforms.
    return BboxParams(
        format="yolo",
        label_fields=["class_labels", "indices"],
        min_area=0,
        min_visibility=0.0,
    )


def _make_transform_args(
    **overrides: Any,
) -> LTDETRInstanceSegmentationTransformArgs:
    defaults: dict[str, Any] = dict(
        channel_drop=None,
        num_channels=3,
        photometric_distort=None,
        random_zoom_out=None,
        random_iou_crop=None,
        random_flip=None,
        random_rotate_90=None,
        random_rotate=None,
        image_size=_get_image_size(),
        bbox_params=_get_bbox_params(),
        resize=_get_resize_args(),
        normalize=None,
        mosaic=None,
    )
    defaults.update(overrides)
    return LTDETRInstanceSegmentationTransformArgs(**defaults)


def _make_input(
    *,
    num_instances: int,
    height: int = 128,
    width: int = 128,
) -> LTDETRInstanceSegmentationTransformInput:
    image: NDArray[np.uint8] = np.random.randint(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    if num_instances == 0:
        bboxes = np.zeros((0, 4), dtype=np.float64)
        class_labels = np.zeros((0,), dtype=np.int64)
        binary_masks = np.zeros((0, height, width), dtype=np.uint8)
    else:
        # YOLO normalized format: (cx, cy, w, h) in [0, 1].
        bboxes = np.array([[0.3, 0.3, 0.2, 0.2]] * num_instances, dtype=np.float64)
        class_labels = np.arange(1, num_instances + 1, dtype=np.int64)
        binary_masks = np.zeros((num_instances, height, width), dtype=np.uint8)
        # Give each instance a distinct non-empty mask region.
        for i in range(num_instances):
            binary_masks[i, 10 : 30 + i, 10 : 30 + i] = 1
    return {
        "image": image,
        "binary_masks": binary_masks,
        "bboxes": bboxes,
        "class_labels": class_labels,
    }


class TestLTDETRInstanceSegmentationTransform:
    @pytest.mark.parametrize(
        "overrides",
        [
            {},
            {"random_flip": _get_random_flip_args()},
            {"random_iou_crop": _get_random_iou_crop_args()},
            {"random_zoom_out": _get_random_zoom_out_args()},
            {"random_rotate_90": _get_random_rotate_90_args()},
            {"random_rotate": _get_random_rotate_args()},
            {"photometric_distort": _get_photometric_distort_args()},
            {"channel_drop": _get_channel_drop_args()},
            {"normalize": _get_normalize_args()},
            {"mosaic": _get_mosaic_args()},
            {
                "random_flip": _get_random_flip_args(),
                "random_iou_crop": _get_random_iou_crop_args(),
                "random_zoom_out": _get_random_zoom_out_args(),
                "photometric_distort": _get_photometric_distort_args(),
                "normalize": _get_normalize_args(),
            },
        ],
    )
    def test___call__output_contract(self, overrides: dict[str, Any]) -> None:
        transform_args = _make_transform_args(**overrides)
        transform_args.resolve_auto(model_init_args={})
        transform = LTDETRInstanceSegmentationTransform(transform_args)

        tr_input = _make_input(num_instances=2)
        tr_output = transform(tr_input)

        # Instance segmentation returns tensors for image and masks, but keeps
        # bboxes/class_labels as numpy arrays.
        assert isinstance(tr_output["image"], Tensor)
        assert tr_output["image"].dtype == torch.float32
        assert isinstance(tr_output["binary_masks"], Tensor)
        assert tr_output["binary_masks"].dtype == torch.int
        assert isinstance(tr_output["bboxes"], np.ndarray)
        assert tr_output["bboxes"].dtype == np.float64
        assert isinstance(tr_output["class_labels"], np.ndarray)
        assert tr_output["class_labels"].dtype == np.int64

        # Masks stay aligned with the surviving boxes and match the image size.
        num_kept = tr_output["bboxes"].shape[0]
        _, height, width = tr_output["binary_masks"].shape
        assert tr_output["binary_masks"].shape[0] == num_kept
        assert tr_output["class_labels"].shape[0] == num_kept
        assert tr_output["image"].shape[-2:] == (height, width)

    def test_masks_track_bboxes(self) -> None:
        # A crop can drop or reorder instances; the mask count must always equal the
        # surviving bbox count so masks stay aligned with their boxes.
        np.random.seed(0)
        transform_args = _make_transform_args(
            random_iou_crop=_get_random_iou_crop_args(),
            random_flip=_get_random_flip_args(),
        )
        transform_args.resolve_auto(model_init_args={})
        transform = LTDETRInstanceSegmentationTransform(transform_args)

        for _ in range(5):
            tr_output = transform(_make_input(num_instances=4))
            assert tr_output["binary_masks"].shape[0] == tr_output["bboxes"].shape[0]
            assert tr_output["class_labels"].shape[0] == tr_output["bboxes"].shape[0]

    def test_empty_masks(self) -> None:
        transform_args = _make_transform_args()
        transform_args.resolve_auto(model_init_args={})
        transform = LTDETRInstanceSegmentationTransform(transform_args)

        tr_output = transform(_make_input(num_instances=0))

        assert isinstance(tr_output["binary_masks"], Tensor)
        assert tr_output["binary_masks"].dtype == torch.int
        assert tr_output["binary_masks"].shape[0] == 0
        # The empty masks tensor still carries the resized spatial dimensions.
        assert tr_output["binary_masks"].shape[-2:] == tr_output["image"].shape[-2:]

    def test_mosaic_mask_aware(self) -> None:
        np.random.seed(0)
        transform_args = _make_transform_args(mosaic=_get_mosaic_args())
        transform_args.resolve_auto(model_init_args={})
        transform = LTDETRInstanceSegmentationTransform(transform_args)

        tr_output = transform(_make_input(num_instances=3))

        assert isinstance(tr_output["image"], Tensor)
        assert tr_output["binary_masks"].shape[0] == tr_output["bboxes"].shape[0]


class TestLTDETRInstanceSegmentationTrainTransformArgs:
    def test_resolve_auto_rejects_photometric_distort_for_non_rgb(self) -> None:
        transform_args = LTDETRInstanceSegmentationTrainTransformArgs(
            channel_drop=ChannelDropArgs(
                num_channels_keep=2,
                weight_drop=(1.0, 1.0, 0.0, 0.0),
            ),
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "photometric_distort only supports RGB images but num_channels is 2"
            ),
        ):
            transform_args.resolve_auto(model_init_args={})


def _make_dataset_item(
    *,
    image_path: str,
    num_instances: int,
    channels: int = 3,
    height: int = 8,
    width: int = 8,
) -> InstanceSegmentationDatasetItem:
    image = torch.rand(channels, height, width)
    bboxes = torch.tensor(
        [[0.3, 0.3, 0.2, 0.2]] * num_instances, dtype=torch.float32
    ).reshape(num_instances, 4)
    classes = torch.arange(1, num_instances + 1, dtype=torch.int64)
    masks = torch.zeros(num_instances, height, width, dtype=torch.int)
    for i in range(num_instances):
        masks[i, 0, 0] = 1
    return {
        "image_path": image_path,
        "image": image,
        "binary_masks": {"masks": masks, "labels": classes},
        "bboxes": bboxes,
        "classes": classes,
    }


class TestLTDETRInstanceSegmentationCollateFunction:
    def test__call__train(self) -> None:
        # mixup disabled so the batch structure is asserted without mixing.
        transform_args = LTDETRInstanceSegmentationTrainTransformArgs(mixup=None)
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRInstanceSegmentationCollateFunction(
            split="train", transform_args=transform_args
        )

        batch = [
            _make_dataset_item(image_path="img1.png", num_instances=2),
            _make_dataset_item(image_path="img2.png", num_instances=3),
        ]
        out = collate_fn(batch)

        assert isinstance(out["image"], Tensor)
        assert out["image"].shape == (2, 3, 8, 8)
        assert isinstance(out["bboxes"], list)
        assert isinstance(out["classes"], list)
        assert isinstance(out["binary_masks"], list)
        assert [b.shape[0] for b in out["bboxes"]] == [2, 3]
        assert all(bm["masks"].dtype == torch.bool for bm in out["binary_masks"])
        assert out["image_path"] == ["img1.png", "img2.png"]

    def test__call__val_split(self) -> None:
        transform_args = LTDETRInstanceSegmentationTrainTransformArgs(mixup=None)
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRInstanceSegmentationCollateFunction(
            split="val", transform_args=transform_args
        )

        batch = [
            _make_dataset_item(image_path="img1.png", num_instances=2),
            _make_dataset_item(image_path="img2.png", num_instances=1),
        ]
        out = collate_fn(batch)

        # Validation keeps images as a list (no stacking / mixup).
        assert isinstance(out["image"], list)
        assert all(isinstance(img, Tensor) for img in out["image"])
        assert [b.shape[0] for b in out["bboxes"]] == [2, 1]

    def test_mixup_concatenates_instances(self) -> None:
        transform_args = LTDETRInstanceSegmentationTrainTransformArgs(
            mixup=LTDETRInstanceSegmentationMixUpArgs(
                prob=1.0, step_start=0, step_stop=None
            ),
        )
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRInstanceSegmentationCollateFunction(
            split="train", transform_args=transform_args
        )

        batch = [
            _make_dataset_item(image_path="img1.png", num_instances=2),
            _make_dataset_item(image_path="img2.png", num_instances=3),
        ]
        out = collate_fn(batch)

        # Each image gains its rolled partner's instances: 2+3 and 3+2.
        assert [b.shape[0] for b in out["bboxes"]] == [5, 5]
        assert [c.shape[0] for c in out["classes"]] == [5, 5]
        assert [bm["masks"].shape[0] for bm in out["binary_masks"]] == [5, 5]
        assert isinstance(out["image"], Tensor)
        assert out["image"].shape == (2, 3, 8, 8)

    def test_requires_dataloader_reinitialization(self) -> None:
        transform_args = LTDETRInstanceSegmentationTrainTransformArgs(
            mixup=LTDETRInstanceSegmentationMixUpArgs(
                prob=1.0, step_start=1, step_stop=2
            ),
        )
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRInstanceSegmentationCollateFunction(
            split="train", transform_args=transform_args
        )

        assert collate_fn.requires_dataloader_reinitialization() is False

        # step 1: mixup activates
        collate_fn.set_step(1)
        assert collate_fn.requires_dataloader_reinitialization() is True
        collate_fn.mark_dataloader_as_reinitialized()
        assert collate_fn.requires_dataloader_reinitialization() is False

        # step 2: mixup deactivates
        collate_fn.set_step(2)
        assert collate_fn.requires_dataloader_reinitialization() is True
        collate_fn.mark_dataloader_as_reinitialized()
        assert collate_fn.requires_dataloader_reinitialization() is False
