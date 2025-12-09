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
import torch
from albumentations import BboxParams
from numpy.typing import NDArray

from lightly_train._data.task_batch_collation import ObjectDetectionCollateFunction
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
    ObjectDetectionTransformInput,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
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
        step_seeding=True,
        seed_offset=0,
    )


def _get_resize_args() -> ResizeArgs:
    return ResizeArgs(
        height=64,
        width=64,
    )


def _get_normalize_args() -> NormalizeArgs:
    return NormalizeArgs()


def _get_image_size() -> tuple[int, int]:
    return (64, 64)


PossibleArgsTuple = (
    [None, _get_channel_drop_args()],
    [None, _get_photometric_distort_args()],
    [None, _get_random_zoom_out_args()],
    [None, _get_random_iou_crop_args()],
    [None, _get_random_flip_args()],
    # TODO: Lionel (09/25) Add StopPolicyArgs test cases.
    [None, _get_scale_jitter_args()],
    [None, _get_resize_args()],
    [None, _get_normalize_args()],
)

possible_tuples = list(itertools.product(*PossibleArgsTuple))


class TestObjectDetectionTransform:
    @pytest.mark.parametrize(
        "channel_drop, photometric_distort, random_zoom_out, random_iou_crop, random_flip, scale_jitter, resize, normalize",
        possible_tuples,
    )
    def test___all_args_combinations(
        self,
        channel_drop: ChannelDropArgs | None,
        photometric_distort: RandomPhotometricDistortArgs | None,
        random_zoom_out: RandomZoomOutArgs | None,
        random_flip: RandomFlipArgs | None,
        scale_jitter: ScaleJitterArgs | None,
        resize: ResizeArgs | None,
        random_iou_crop: RandomIoUCropArgs | None,
        normalize: NormalizeArgs | None,
    ) -> None:
        image_size = _get_image_size()
        bbox_params = _get_bbox_params()
        stop_policy = None  # TODO: Lionel (09/25) Pass as function argument.
        transform_args = ObjectDetectionTransformArgs(
            channel_drop=channel_drop,
            num_channels=3,
            photometric_distort=photometric_distort,
            random_zoom_out=random_zoom_out,
            random_iou_crop=random_iou_crop,
            random_flip=random_flip,
            image_size=image_size,
            bbox_params=bbox_params,
            stop_policy=stop_policy,
            resize=resize,
            scale_jitter=scale_jitter,
            normalize=normalize,
        )
        transform_args.resolve_auto(model_init_args={})
        transform = ObjectDetectionTransform(transform_args)

        # Create a synthetic image and bounding boxes.
        num_channels = transform_args.num_channels
        assert num_channels != "auto"
        img: NDArray[np.uint8] = np.random.randint(
            0, 256, (128, 128, num_channels), dtype=np.uint8
        )
        bboxes = np.array([[10, 10, 50, 50]], dtype=np.float64)
        class_labels = np.array([1], dtype=np.int64)

        tr_input: ObjectDetectionTransformInput = {
            "image": img,
            "bboxes": bboxes,
            "class_labels": class_labels,
        }
        tr_output = transform(tr_input)
        assert isinstance(tr_output, dict)
        out_img = tr_output["image"]
        assert isinstance(out_img, torch.Tensor)
        assert out_img.dtype == torch.float32
        assert "bboxes" in tr_output
        assert "class_labels" in tr_output

    def test__collation(self) -> None:
        transform_args = ObjectDetectionTransformArgs(
            channel_drop=_get_channel_drop_args(),
            num_channels=3,
            photometric_distort=_get_photometric_distort_args(),
            random_zoom_out=_get_random_zoom_out_args(),
            random_iou_crop=_get_random_iou_crop_args(),
            random_flip=_get_random_flip_args(),
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
            "image": torch.randn(3, 128, 128),
            "bboxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "classes": torch.tensor([1]),
            "original_size": (128, 128),
        }
        sample2: ObjectDetectionDatasetItem = {
            "image_path": "img2.png",
            "image": torch.randn(3, 64, 64),
            "bboxes": torch.tensor([[20.0, 20.0, 40.0, 40.0]]),
            "classes": torch.tensor([2]),
            "original_size": (64, 64),
        }
        batch = [sample1, sample2]

        out = collate_fn(batch)
        assert isinstance(out, dict)
