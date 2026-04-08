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

from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    DINOv3LTDETRObjectDetectionCopyBlendArgs,
    DINOv3LTDETRObjectDetectionMixUpArgs,
    DINOv3LTDETRObjectDetectionScaleJitterArgs,
    DINOv3LTDETRObjectDetectionTrainTransformArgs,
)
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionCollateFunction,
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
    ObjectDetectionTransformInput,
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
    StopPolicyArgs,
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
        prob=0.5,
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


def _get_bbox_params() -> BboxParams:
    return BboxParams(
        format="pascal_voc",
        label_fields=["class_labels"],
        min_area=0,
        min_visibility=0.0,
    )


def _get_stop_policy_args() -> StopPolicyArgs:
    return StopPolicyArgs(
        stop_step=500_000,
        ops={ChannelDrop},
    )


def _get_scale_jitter_args() -> ScaleJitterArgs:
    return ScaleJitterArgs(
        sizes=None,
        min_scale=0.76,
        max_scale=1.27,
        num_scales=13,
        prob=1.0,
        divisible_by=14,
    )


def _get_resize_args() -> ResizeArgs:
    return ResizeArgs(
        height=64,
        width=64,
    )


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


PossibleArgsTuple = (
    [None, _get_channel_drop_args()],
    [None, _get_photometric_distort_args()],
    [None, _get_random_zoom_out_args()],
    [None, _get_random_iou_crop_args()],
    [None, _get_random_flip_args()],
    [None, _get_random_rotate_90_args()],
    [None, _get_random_rotate_args()],
    # TODO: Lionel (09/25) Add StopPolicyArgs test cases.
    [None, _get_scale_jitter_args()],
    [None, _get_resize_args()],
    [None, _get_normalize_args()],
    [None, _get_mosaic_args()],
)

possible_tuples = list(itertools.product(*PossibleArgsTuple))


class TestObjectDetectionTransform:
    @pytest.mark.parametrize(
        "channel_drop, photometric_distort, random_zoom_out, random_iou_crop, random_flip, random_rotate_90, random_rotate, scale_jitter, resize, normalize, mosaic",
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
        # MosaicTransform expects YOLO normalized bboxes (matching production configs).
        if mosaic is not None:
            bbox_params = BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_area=0,
                min_visibility=0.0,
            )
        else:
            bbox_params = _get_bbox_params()
        stop_policy = None  # TODO: Lionel (09/25) Pass as function argument.
        transform_args = ObjectDetectionTransformArgs(
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
            stop_policy=stop_policy,
            resize=resize,
            scale_jitter=scale_jitter,
            normalize=normalize,
            mosaic=mosaic,
        )
        transform_args.resolve_auto(model_init_args={})
        transform = ObjectDetectionTransform(transform_args)

        # Create a synthetic image and bounding boxes.
        num_channels = transform_args.num_channels
        assert num_channels != "auto"
        img: NDArray[np.uint8] = np.random.randint(
            0, 256, (128, 128, num_channels), dtype=np.uint8
        )
        if mosaic is not None:
            # YOLO normalized format: (cx, cy, w, h) in [0, 1].
            bboxes = np.array([[0.234, 0.234, 0.3125, 0.3125]], dtype=np.float64)
        else:
            bboxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        class_labels = np.array([1], dtype=np.int64)

        tr_input: ObjectDetectionTransformInput = {
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

    def test_requires_dataloader_reinitialization(self) -> None:
        transform_args = ObjectDetectionTransformArgs(
            channel_drop=None,
            num_channels=3,
            photometric_distort=None,
            random_zoom_out=_get_random_zoom_out_args(),
            random_iou_crop=_get_random_iou_crop_args(),
            random_flip=_get_random_flip_args(),
            random_rotate_90=None,
            random_rotate=None,
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            stop_policy=None,
            resize=_get_resize_args(),
            scale_jitter=None,
            normalize=None,
            mosaic=MosaicArgs(
                prob=0.5,
                step_start=2,
                step_stop=5,
                output_size=320,
                max_size=None,
                rotation_range=10.0,
                translation_range=(0.1, 0.1),
                scaling_range=(0.5, 1.5),
                fill_value=0,
                max_cached_images=50,
                random_pop=True,
            ),
        )
        transform_args.resolve_auto(model_init_args={})
        transform = ObjectDetectionTransform(transform_args)

        # step 0: no reinit needed
        assert transform.requires_dataloader_reinitialization() is False

        # step 2: mosaic activates -> reinit True, mark, then False
        transform.set_step(2)
        assert transform.requires_dataloader_reinitialization() is True
        transform.mark_dataloader_as_reinitialized()
        assert transform.requires_dataloader_reinitialization() is False

        # step 4: still active -> no reinit
        transform.set_step(4)
        assert transform.requires_dataloader_reinitialization() is False

        # step 5: mosaic deactivates -> reinit True, mark, then False
        transform.set_step(5)
        assert transform.requires_dataloader_reinitialization() is True
        transform.mark_dataloader_as_reinitialized()
        assert transform.requires_dataloader_reinitialization() is False


class TestObjectDetectionCollateFunction:
    def test__call__(self) -> None:
        transform_args = ObjectDetectionTransformArgs(
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
            stop_policy=_get_stop_policy_args(),
            scale_jitter=_get_scale_jitter_args(),
            resize=_get_resize_args(),
            normalize=_get_normalize_args(),
        )
        transform_args.resolve_auto(model_init_args={})
        collate_fn = ObjectDetectionCollateFunction(
            split="train", transform_args=transform_args
        )

        sample1: ObjectDetectionDatasetItem = {
            "image_path": "img1.png",
            "image": np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            "bboxes": np.array([[10.0, 10.0, 50.0, 50.0]]),
            "classes": np.array([1], dtype=np.int64),
            "original_size": (128, 128),
        }
        sample2: ObjectDetectionDatasetItem = {
            "image_path": "img2.png",
            "image": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            "bboxes": np.array([[20.0, 20.0, 40.0, 40.0]]),
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
        transform_args = DINOv3LTDETRObjectDetectionTrainTransformArgs(
            image_size=_get_image_size(),
            bbox_params=_get_bbox_params(),
            mixup=DINOv3LTDETRObjectDetectionMixUpArgs(
                prob=1.0,
                step_start=1,
                step_stop=2,
            ),
            copyblend=DINOv3LTDETRObjectDetectionCopyBlendArgs(
                prob=1.0,
                step_start=2,
                step_stop=4,
                area_threshold=1,
                num_objects=1,
                expand_ratios=(0.1, 0.25),
            ),
            scale_jitter=DINOv3LTDETRObjectDetectionScaleJitterArgs(
                step_stop=3,
                sizes=[(32, 32)],
                min_scale=None,
                max_scale=None,
                num_scales=None,
                prob=1.0,
                divisible_by=None,
            ),
        )
        transform_args.resolve_auto(model_init_args={})

        collate_fn = ObjectDetectionCollateFunction(
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
