#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import itertools

import numpy as np
import pytest
from albumentations import BboxParams
from numpy.typing import NDArray
from torch import Tensor

from lightly_train._task_models.ltdetr_object_detection.transforms import (
    LTDETRObjectDetectionCopyBlendArgs,
    LTDETRObjectDetectionMixUpArgs,
    LTDETRObjectDetectionScaleJitterArgs,
    LTDETRObjectDetectionTrainTransformArgs,
    LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._transforms.ltdetr_transforms.object_detection import (
    LTDETRObjectDetectionCollateFunction,
    LTDETRObjectDetectionTransform,
    LTDETRObjectDetectionTransformArgs,
    LTDETRObjectDetectionTransformInput,
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
    ScaleJitterArgs,
)
from lightly_train.types import ObjectDetectionDatasetItem


def _get_channel_drop_args() -> ChannelDropArgs:
    return ChannelDropArgs(
        num_channels_keep=3,
        weight_drop=(1.0, 1.0, 0.0, 0.0),
    )


def _get_random_flip_args() -> RandomFlipArgs:
    return RandomFlipArgs(horizontal_prob=0.5, vertical_prob=0.5)


def _get_random_rotate_90_args() -> RandomRotate90Args:
    return RandomRotate90Args(prob=0.3)


def _get_random_rotate_args() -> RandomRotationArgs:
    return RandomRotationArgs(prob=0.4, degrees=30.0)


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


def _get_resize_args() -> ResizeArgs:
    return ResizeArgs(
        height=64,
        width=64,
    )


def _get_scale_jitter_args(
    *,
    step_stop: int | None = None,
) -> LTDETRObjectDetectionScaleJitterArgs:
    return LTDETRObjectDetectionScaleJitterArgs(
        sizes=None,
        min_scale=0.76,
        max_scale=1.27,
        num_scales=13,
        prob=1.0,
        divisible_by=14,
        step_stop=step_stop,
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


def _get_normalize_args() -> NormalizeArgs:
    return NormalizeArgs()


def _get_mixup_args(
    *,
    step_start: int,
    step_stop: int,
) -> LTDETRObjectDetectionMixUpArgs:
    return LTDETRObjectDetectionMixUpArgs(
        prob=1.0,
        step_start=step_start,
        step_stop=step_stop,
    )


def _get_copyblend_args(
    *,
    step_start: int,
    step_stop: int,
) -> LTDETRObjectDetectionCopyBlendArgs:
    return LTDETRObjectDetectionCopyBlendArgs(
        prob=1.0,
        step_start=step_start,
        step_stop=step_stop,
        area_threshold=1,
        num_objects=1,
        expand_ratios=(0.1, 0.25),
    )


def _get_image_size() -> tuple[int, int]:
    return (64, 64)


def _get_bbox_params() -> BboxParams:
    return BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_area=0,
        min_visibility=0.0,
    )


PossibleArgsTuple = (
    [None, _get_channel_drop_args()],
    [None, _get_photometric_distort_args()],
    [None, _get_random_zoom_out_args()],
    [None, _get_random_iou_crop_args()],
    [None, _get_random_flip_args()],
    [None, _get_random_rotate_90_args()],
    [None, _get_random_rotate_args()],
    [None, _get_resize_args()],
    [None, _get_normalize_args()],
    [None, _get_scale_jitter_args()],
    [None, _get_mosaic_args()],
)

possible_tuples = list(itertools.product(*PossibleArgsTuple))


class TestObjectDetectionTransform:
    @pytest.mark.parametrize(
        "channel_drop, photometric_distort, random_zoom_out, random_iou_crop, random_flip, random_rotate_90, random_rotate, resize, normalize, scale_jitter, mosaic",
        possible_tuples,
    )
    def test___all_args_combinations(
        self,
        channel_drop: ChannelDropArgs | None,
        photometric_distort: RandomPhotometricDistortArgs | None,
        random_zoom_out: RandomZoomOutArgs | None,
        random_flip: RandomFlipArgs | None,
        random_rotate_90: RandomRotate90Args | None,
        random_rotate: RandomRotationArgs | None,
        scale_jitter: ScaleJitterArgs | None,
        resize: ResizeArgs | None,
        random_iou_crop: RandomIoUCropArgs | None,
        normalize: NormalizeArgs | None,
        mosaic: MosaicArgs | None,
    ) -> None:
        image_size = _get_image_size()
        bbox_params = _get_bbox_params()

        transform_args = LTDETRObjectDetectionTransformArgs(
            channel_drop=channel_drop,
            num_channels=3,
            photometric_distort=photometric_distort,
            random_zoom_out=random_zoom_out,
            random_iou_crop=random_iou_crop,
            random_flip=random_flip,
            random_rotate_90=random_rotate_90,
            random_rotate=random_rotate,
            image_size=image_size,
            bbox_params=bbox_params,
            resize=resize,
            scale_jitter=scale_jitter,
            normalize=normalize,
            mosaic=mosaic,
        )
        transform_args.resolve_auto(model_init_args={})
        transform = LTDETRObjectDetectionTransform(transform_args)

        # Create a synthetic image and bounding boxes.
        num_channels = transform_args.num_channels
        assert num_channels != "auto"
        img: NDArray[np.uint8] = np.random.randint(
            0, 256, (128, 128, num_channels), dtype=np.uint8
        )
        # YOLO normalized format: (cx, cy, w, h) in [0, 1].
        bboxes = np.array([[0.234375, 0.234375, 0.3125, 0.3125]], dtype=np.float64)
        class_labels = np.array([1], dtype=np.int64)

        tr_input: LTDETRObjectDetectionTransformInput = {
            "image": img,
            "bboxes": bboxes,
            "class_labels": class_labels,
        }
        tr_output = transform(tr_input)
        assert tr_output["image"].dtype == np.float32
        assert "bboxes" in tr_output
        assert tr_output["bboxes"].dtype == np.float64
        assert "class_labels" in tr_output
        assert tr_output["class_labels"].dtype == np.int64


class TestObjectDetectionCollateFunction:
    def test__call__(self) -> None:
        transform_args = LTDETRObjectDetectionTransformArgs(
            channel_drop=_get_channel_drop_args(),
            num_channels=3,
            photometric_distort=_get_photometric_distort_args(),
            random_zoom_out=_get_random_zoom_out_args(),
            random_iou_crop=_get_random_iou_crop_args(),
            random_flip=_get_random_flip_args(),
            random_rotate_90=_get_random_rotate_90_args(),
            random_rotate=_get_random_rotate_args(),
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            scale_jitter=_get_scale_jitter_args(),
            resize=_get_resize_args(),
            normalize=_get_normalize_args(),
        )
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRObjectDetectionCollateFunction(
            split="train", transform_args=transform_args
        )

        sample1: ObjectDetectionDatasetItem = {
            "image_path": "img1.png",
            "image": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            "bboxes": np.array([[0.234375, 0.234375, 0.3125, 0.3125]]),
            "classes": np.array([1], dtype=np.int64),
            "original_size": (128, 128),
        }
        sample2: ObjectDetectionDatasetItem = {
            "image_path": "img2.png",
            "image": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            "bboxes": np.array([[0.46875, 0.46875, 0.3125, 0.3125]]),
            "classes": np.array([2], dtype=np.int64),
            "original_size": (64, 64),
        }
        batch = [sample1, sample2]

        out = collate_fn(batch)
        assert isinstance(out["image"], Tensor)
        assert isinstance(out["bboxes"], list)
        assert isinstance(out["classes"], list)
        assert isinstance(out["original_size"], list)
        assert all(isinstance(bbox, Tensor) for bbox in out["bboxes"])
        assert all(isinstance(classes, Tensor) for classes in out["classes"])
        assert out["original_size"] == [(128, 128), (64, 64)]

    def test_requires_dataloader_reinitialization(self) -> None:
        transform_args = LTDETRObjectDetectionTrainTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            mixup=_get_mixup_args(
                step_start=1,
                step_stop=2,
            ),
            copyblend=_get_copyblend_args(
                step_start=2,
                step_stop=4,
            ),
            scale_jitter=_get_scale_jitter_args(
                step_stop=3,
            ),
        )
        transform_args.resolve_auto(model_init_args={})

        collate_fn = LTDETRObjectDetectionCollateFunction(
            split="train",
            transform_args=transform_args,
        )

        assert collate_fn.requires_dataloader_reinitialization() is False

        collate_fn.set_step(1)
        assert collate_fn.requires_dataloader_reinitialization() is True
        assert collate_fn.requires_dataloader_reinitialization() is True
        collate_fn.mark_dataloader_as_reinitialized()
        assert collate_fn.requires_dataloader_reinitialization() is False

        collate_fn.set_step(2)
        assert collate_fn.requires_dataloader_reinitialization() is True
        collate_fn.mark_dataloader_as_reinitialized()
        assert collate_fn.requires_dataloader_reinitialization() is False

        collate_fn.set_step(3)
        assert collate_fn.requires_dataloader_reinitialization() is True
        collate_fn.mark_dataloader_as_reinitialized()
        assert collate_fn.requires_dataloader_reinitialization() is False

        collate_fn.set_step(4)
        assert collate_fn.requires_dataloader_reinitialization() is True
        collate_fn.mark_dataloader_as_reinitialized()
        assert collate_fn.requires_dataloader_reinitialization() is False


class TestObjectDetectionTransformBboxFilter:
    """The per-sample transform no longer filters by ``min_bbox_size_px``."""

    def test_does_not_drop_sub_min_size_boxes(self) -> None:
        # min_bbox_size_px is only enforced in the collate function now, so a
        # sub-minimum box must survive the per-sample transform untouched.
        transform_args = LTDETRObjectDetectionTrainTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            scale_jitter=None,
            mosaic=None,
            min_bbox_size_px=4.0,
        )
        transform_args.resolve_auto(model_init_args={})
        transform = LTDETRObjectDetectionTransform(transform_args)

        # Boxes are normalized to the original 128x128 image.
        bboxes = np.array(
            [
                [0.2, 0.2, 0.3, 0.3],  # ~38x38
                [0.5, 0.5, 0.0001, 0.3],  # ~0x38 -> sub-min width
                [0.8, 0.8, 0.3, 0.0001],  # 38x0 -> sub-min height
                [0.5, 0.5, 0.2, 0.2],  # 25x25
            ],
            dtype=np.float64,
        )
        class_labels = np.array([1, 2, 3, 4], dtype=np.int64)

        tr_input: LTDETRObjectDetectionTransformInput = {
            "image": np.full((128, 128, 3), 127, dtype=np.uint8),
            "bboxes": bboxes,
            "class_labels": class_labels,
        }
        tr_output = transform(tr_input)

        assert tr_output["bboxes"].shape[0] == 4
        assert sorted(tr_output["class_labels"].tolist()) == [1, 2, 3, 4]


class TestObjectDetectionCollateBboxFilter:
    """Behavior tests for the post-batch-transform bbox size filter."""

    def test_filters_sub_min_size_after_resize(self) -> None:
        # Without scale_jitter, filtering still applies using the final
        # (post-resize) image size.
        transform_args = LTDETRObjectDetectionTrainTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            scale_jitter=None,
            mosaic=None,
            min_bbox_size_px=4.0,
        )
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRObjectDetectionCollateFunction(
            split="train",
            transform_args=transform_args,
        )

        sample: ObjectDetectionDatasetItem = {
            "image_path": "img.png",
            "image": np.full((128, 128, 3), 127, dtype=np.uint8),
            "bboxes": np.array(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.2, 0.2, 0.0001, 0.0001],
                    [0.8, 0.8, 0.0001, 0.0001],
                ]
            ),
            "classes": np.array([1, 2, 3], dtype=np.int64),
            "original_size": (128, 128),
        }

        out = collate_fn([sample])

        assert out["bboxes"][0].shape[0] == 1
        assert out["classes"][0].tolist() == [1]  # type: ignore[union-attr]  # numpy fancy-index.

    def test_filters_sub_min_size_after_scale_jitter(self) -> None:
        # After scale_jitter downsamples images to ~480x480, boxes with
        # normalized width/height < 4/480 are dropped.
        transform_args = LTDETRObjectDetectionTrainTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            scale_jitter=LTDETRObjectDetectionScaleJitterArgs(
                sizes=[(480, 480)],
                min_scale=None,
                max_scale=None,
                num_scales=None,
                prob=1.0,
                divisible_by=None,
                step_stop=None,
            ),
            mosaic=None,
            min_bbox_size_px=4.0,
        )
        transform_args.resolve_auto(model_init_args={})
        collate_fn = LTDETRObjectDetectionCollateFunction(
            split="train",
            transform_args=transform_args,
        )

        sample: ObjectDetectionDatasetItem = {
            "image_path": "img.png",
            "image": np.full((128, 128, 3), 127, dtype=np.uint8),
            # Valid 25x25 box and two sub-min 1 px boxes after 480 resize.
            "bboxes": np.array(
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.2, 0.2, 0.0001, 0.0001],
                    [0.8, 0.8, 0.0001, 0.0001],
                ]
            ),
            "classes": np.array([1, 2, 3], dtype=np.int64),
            "original_size": (128, 128),
        }

        out = collate_fn([sample])

        # Only the 25x25 box stays: 1/480 and 0.01/480 are both below 4 px.
        assert out["bboxes"][0].shape[0] == 1
        assert out["classes"][0].tolist() == [1]  # type: ignore[union-attr]  # numpy fancy-index.


class TestMinBboxSizePxDefaults:
    """The 4 px guard must only be on by default for LT-DETR training."""

    def test_base_args_default_to_disabled(self) -> None:
        assert (
            LTDETRObjectDetectionTransformArgs.model_fields["min_bbox_size_px"].default
            == 0.0
        )

    def test_train_args_default_to_four_pixels(self) -> None:
        transform_args = LTDETRObjectDetectionTrainTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
        )
        assert transform_args.min_bbox_size_px == 4.0

    def test_val_args_default_to_disabled(self) -> None:
        transform_args = LTDETRObjectDetectionValTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
        )
        assert transform_args.min_bbox_size_px == 0.0
