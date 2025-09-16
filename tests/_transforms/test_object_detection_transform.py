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
from albumentations import BboxParams, ChannelDrop
from numpy.typing import NDArray

from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
    ObjectDetectionTransformInput,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    StopPolicyArgs,
)


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


def _get_image_size() -> tuple[int, int]:
    return (64, 64)


PossibleArgsTuple = (
    [None, _get_channel_drop_args()],
    [None, _get_photometric_distort_args()],
    [None, _get_random_zoom_out_args()],
    [None, _get_random_flip_args()],
    [None, _get_stop_policy_args()],
)

possible_tuples = list(itertools.product(*PossibleArgsTuple))


class TestObjectDetectionTransform:
    @pytest.mark.parametrize(
        "channel_drop, photometric_distort, random_zoom_out, random_flip, stop_policy",
        possible_tuples,
    )
    def test___all_args_combinations(
        self,
        channel_drop: ChannelDropArgs | None,
        photometric_distort: RandomPhotometricDistortArgs | None,
        random_zoom_out: RandomZoomOutArgs | None,
        random_flip: RandomFlipArgs | None,
        stop_policy: StopPolicyArgs | None,
    ) -> None:
        image_size = _get_image_size()
        bbox_params = _get_bbox_params()
        transform_args = ObjectDetectionTransformArgs(
            channel_drop=channel_drop,
            num_channels="auto",
            photometric_distort=photometric_distort,
            random_zoom_out=random_zoom_out,
            random_flip=random_flip,
            image_size=image_size,
            bbox_params=bbox_params,
            stop_policy=stop_policy,
        )
        transform_args.resolve_auto()
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
        assert out_img.shape[1:] == image_size
        assert out_img.dtype == torch.float32
        assert "bboxes" in tr_output
        assert "class_labels" in tr_output
