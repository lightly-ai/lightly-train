#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import pytest
from albumentations import BboxParams

from lightly_train._task_models.ltdetr_object_detection.transforms import (
    LTDETRObjectDetectionTrainTransformArgs,
)
from lightly_train._transforms.ltdetr_transforms.object_detection import (
    LTDETRObjectDetectionTransform,
    LTDETRObjectDetectionTransformArgs,
)
from lightly_train._transforms.ltdetr_transforms.utils import (
    resolve_ltdetr_step_schedule_for_augmentation,
)
from lightly_train._transforms.transform import (
    MosaicArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ResizeArgs,
)

# The step-scheduling machinery, Compose caching, and dataloader-reinitialization logic
# live in the shared ``_LTDETRTransform`` / ``LTDETRTransformArgs`` base classes and
# behave identically for object detection and instance segmentation. They are therefore
# tested once here, driven through the object detection concrete classes (the thinnest
# concrete wrappers around the internal bases).


def _get_image_size() -> tuple[int, int]:
    return (64, 64)


def _get_bbox_params() -> BboxParams:
    return BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_area=0,
        min_visibility=0.0,
    )


def _get_photometric_distort_args(
    *,
    step_start: int = 0,
    step_stop: int | None = None,
) -> RandomPhotometricDistortArgs:
    return RandomPhotometricDistortArgs(
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
        prob=0.5,
        step_start=step_start,
        step_stop=step_stop,
    )


def _get_random_zoom_out_args(
    *,
    step_start: int = 0,
    step_stop: int | None = None,
) -> RandomZoomOutArgs:
    return RandomZoomOutArgs(
        prob=0.5,
        fill=0.0,
        side_range=(1.0, 1.5),
        step_start=step_start,
        step_stop=step_stop,
    )


def _get_random_iou_crop_args(
    *,
    step_start: int = 0,
    step_stop: int | None = None,
) -> RandomIoUCropArgs:
    return RandomIoUCropArgs(
        min_scale=0.3,
        max_scale=1.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        sampler_options=None,
        crop_trials=40,
        iou_trials=1000,
        prob=1.0,
        step_start=step_start,
        step_stop=step_stop,
    )


def _get_random_flip_args() -> RandomFlipArgs:
    return RandomFlipArgs(horizontal_prob=0.5, vertical_prob=0.5)


def _get_resize_args() -> ResizeArgs:
    return ResizeArgs(
        height=64,
        width=64,
    )


def _get_mosaic_args(
    *,
    step_start: int = 0,
    step_stop: int | None = 100,
) -> MosaicArgs:
    return MosaicArgs(
        prob=1.0,
        step_start=step_start,
        step_stop=step_stop,
        output_size=32,
        max_size=None,
        rotation_range=10.0,
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),
        fill_value=0,
        max_cached_images=50,
        random_pop=True,
    )


def _get_ltdetr_train_transform_args() -> LTDETRObjectDetectionTrainTransformArgs:
    return LTDETRObjectDetectionTrainTransformArgs(
        image_size=_get_image_size(),
        bbox_params=_get_bbox_params(),
    )


@pytest.mark.parametrize(
    ("augmentation_name", "step_start", "step_stop"),
    [
        ("photometric_distort", 100, 300),
        ("random_zoom_out", 100, 300),
        ("random_iou_crop", 100, 300),
        ("copyblend", 100, 300),
        ("mixup", 100, 200),
        ("mosaic", 100, 200),
    ],
)
def test_resolve_ltdetr_step_schedule_for_augmentation__applies_windows(
    augmentation_name: str,
    step_start: int,
    step_stop: int,
) -> None:
    transform_args = _get_ltdetr_train_transform_args()
    transform_args.resolve_auto(model_init_args={})

    # total_steps=300 yields step_start=100, step_flat=200, step_stop=300, so
    # every augmentation receives a non-empty window.
    resolve_ltdetr_step_schedule_for_augmentation(
        args=transform_args,
        total_steps=300,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    aug = getattr(transform_args, augmentation_name)
    assert aug is not None
    assert aug.step_start == step_start
    assert aug.step_stop == step_stop

    assert transform_args.scale_jitter is not None
    assert transform_args.scale_jitter.step_stop == 300


@pytest.mark.parametrize("augmentation_name", ["mixup", "mosaic"])
def test_resolve_ltdetr_step_schedule_for_augmentation__disables_empty_windows(
    augmentation_name: str,
) -> None:
    transform_args = _get_ltdetr_train_transform_args()
    transform_args.resolve_auto(model_init_args={})

    # total_steps=100 yields step_start=0, step_flat=0, step_stop=100, so
    # mixup/mosaic collapse to an empty window and must be disabled.
    resolve_ltdetr_step_schedule_for_augmentation(
        args=transform_args,
        total_steps=100,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    assert getattr(transform_args, augmentation_name) is None

    assert transform_args.scale_jitter is not None
    assert transform_args.scale_jitter.step_stop == 100


@pytest.mark.parametrize(
    "augmentation_name",
    ["photometric_distort", "random_zoom_out", "random_iou_crop", "copyblend"],
)
def test_resolve_ltdetr_step_schedule_for_augmentation__keeps_non_empty_windows(
    augmentation_name: str,
) -> None:
    transform_args = _get_ltdetr_train_transform_args()
    transform_args.resolve_auto(model_init_args={})

    # total_steps=100 yields step_start=0, step_flat=0, step_stop=100, so
    # augmentations using step_start->step_stop still receive a non-empty window.
    resolve_ltdetr_step_schedule_for_augmentation(
        args=transform_args,
        total_steps=100,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    aug = getattr(transform_args, augmentation_name)
    assert aug is not None
    assert aug.step_start == 0
    assert aug.step_stop == 100

    assert transform_args.scale_jitter is not None
    assert transform_args.scale_jitter.step_stop == 100


def test_requires_dataloader_reinitialization() -> None:
    transform_args = LTDETRObjectDetectionTransformArgs(
        channel_drop=None,
        num_channels=3,
        photometric_distort=_get_photometric_distort_args(
            step_start=1,
            step_stop=5,
        ),
        random_zoom_out=_get_random_zoom_out_args(
            step_start=2,
            step_stop=6,
        ),
        random_iou_crop=_get_random_iou_crop_args(
            step_start=3,
            step_stop=7,
        ),
        random_flip=_get_random_flip_args(),
        random_rotate_90=None,
        random_rotate=None,
        image_size=_get_image_size(),
        bbox_params=_get_bbox_params(),
        resize=_get_resize_args(),
        mosaic=_get_mosaic_args(
            step_start=4,
            step_stop=8,
        ),
        normalize=None,
    )
    transform_args.resolve_auto(model_init_args={})
    transform = LTDETRObjectDetectionTransform(transform_args)

    # step 0: no reinit needed
    assert transform.requires_dataloader_reinitialization() is False

    # step 1: photometric_distort activates
    transform.set_step(1)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 2: random_zoom_out activates
    transform.set_step(2)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 3: random_iou_crop activates
    transform.set_step(3)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 4: mosaic activates
    transform.set_step(4)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 5: photometric_distort deactivates
    transform.set_step(5)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 6: random_zoom_out deactivates
    transform.set_step(6)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 7: random_iou_crop deactivates
    transform.set_step(7)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False

    # step 8: mosaic deactivates
    transform.set_step(8)
    assert transform.requires_dataloader_reinitialization() is True
    transform.mark_dataloader_as_reinitialized()
    assert transform.requires_dataloader_reinitialization() is False
