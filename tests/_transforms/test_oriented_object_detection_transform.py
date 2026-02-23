#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest
import torch
from albumentations import BboxParams
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

from lightly_train._data.task_batch_collation import (
    OrientedObjectDetectionCollateFunction,
)
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.oriented_object_detection_transform import (
    OrientedObjectDetectionTransform,
    OrientedObjectDetectionTransformArgs,
    OrientedObjectDetectionTransformInput,
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
    StopPolicyArgs,
)
from lightly_train.types import OrientedObjectDetectionDatasetItem


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
    return RandomRotationArgs(
        prob=0.4, degrees=30.0, interpolation=cv2.INTER_LINEAR
    )  # for OBB , since we use tv transforms, the default cv2.INTER_AREA is not supported.


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


def _get_random_iou_crop_args() -> RandomIoUCropArgs | None:
    return None


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
        min_scale=0.8,
        max_scale=1.2,
        num_scales=5,
        divisible_by=1,
        prob=0.5,
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
    [None, _get_random_rotate_90_args()],
    [None, _get_random_rotate_args()],
    # TODO: Lionel (09/25) Add StopPolicyArgs test cases.
    [None, _get_scale_jitter_args()],
    [None, _get_resize_args()],
    [None, _get_normalize_args()],
)

possible_tuples = list(itertools.product(*PossibleArgsTuple))


class TestObjectDetectionTransform:
    @pytest.mark.parametrize(
        "channel_drop, photometric_distort, random_zoom_out, random_iou_crop, random_flip, random_rotate_90, random_rotate, scale_jitter, resize, normalize",
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
    ) -> None:
        image_size = _get_image_size()
        bbox_params = _get_bbox_params()
        stop_policy = None  # TODO: Lionel (09/25) Pass as function argument.
        transform_args = OrientedObjectDetectionTransformArgs(
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
        )
        transform_args.resolve_auto(model_init_args={})
        transform = OrientedObjectDetectionTransform(transform_args)

        # Create a synthetic image and bounding boxes.
        num_channels = transform_args.num_channels
        assert num_channels != "auto"
        img = Image(torch.randint(0, 256, (num_channels, 128, 128), dtype=torch.uint8))
        bboxes = BoundingBoxes(
            torch.tensor([[10, 10, 50, 50, 45]], dtype=torch.float64),
            format=BoundingBoxFormat.CXCYWHR,
            canvas_size=(128, 128),
        )
        class_labels = np.array([1], dtype=np.int64)

        tr_input: OrientedObjectDetectionTransformInput = {
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
        transform_args = OrientedObjectDetectionTransformArgs(
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
        collate_fn = OrientedObjectDetectionCollateFunction(
            split="train", transform_args=transform_args
        )

        sample1: OrientedObjectDetectionDatasetItem = {
            "image_path": "img1.png",
            "image": torch.randn(3, 128, 128),
            "bboxes": torch.tensor([[10.0, 10.0, 50.0, 50.0, 45]]),
            "classes": torch.tensor([1]),
            "original_size": (128, 128),
        }
        sample2: OrientedObjectDetectionDatasetItem = {
            "image_path": "img2.png",
            "image": torch.randn(3, 64, 64),
            "bboxes": torch.tensor([[20.0, 20.0, 40.0, 40.0, 45]]),
            "classes": torch.tensor([2]),
            "original_size": (64, 64),
        }
        batch = [sample1, sample2]

        out = collate_fn(batch)
        assert isinstance(out, dict)
        assert out["bboxes"][0].shape[-1] == 5
