#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._task_models.ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgsV2,
    DINOv2LTDETRObjectDetectionValTransformArgsV2,
    LTDETRObjectDetectionTrainTransformArgs,
    LTDETRObjectDetectionValTransformArgs,
    LTDETRv2ObjectDetectionTrainTransformArgs,
)
from lightly_train._transforms.transform import NormalizeArgs


@pytest.mark.parametrize(
    "transform_args_cls",
    [
        LTDETRObjectDetectionTrainTransformArgs,
        LTDETRv2ObjectDetectionTrainTransformArgs,
        DINOv2LTDETRObjectDetectionTrainTransformArgsV2,
    ],
)
@pytest.mark.parametrize("num_channels, enabled", [(3, True), (4, False)])
def test_train_transform_args_resolve_auto__rgb_only_transforms(
    transform_args_cls: type[LTDETRObjectDetectionTrainTransformArgs]
    | type[DINOv2LTDETRObjectDetectionTrainTransformArgsV2],
    num_channels: int,
    enabled: bool,
) -> None:
    transform_args = transform_args_cls(num_channels=num_channels)
    transform_args.resolve_auto(model_init_args={})

    assert (transform_args.photometric_distort is not None) is enabled
    assert (transform_args.mosaic is not None) is enabled
    assert isinstance(transform_args.normalize, NormalizeArgs)
    assert len(transform_args.normalize.mean) == num_channels
    assert len(transform_args.normalize.std) == num_channels


@pytest.mark.parametrize(
    "transform_args_cls",
    [
        LTDETRObjectDetectionValTransformArgs,
        DINOv2LTDETRObjectDetectionValTransformArgsV2,
    ],
)
def test_val_transform_args_resolve_auto__normalize_num_channels(
    transform_args_cls: type[LTDETRObjectDetectionValTransformArgs]
    | type[DINOv2LTDETRObjectDetectionValTransformArgsV2],
) -> None:
    transform_args = transform_args_cls(num_channels=4)
    transform_args.resolve_auto(model_init_args={})

    assert isinstance(transform_args.normalize, NormalizeArgs)
    assert len(transform_args.normalize.mean) == 4
    assert len(transform_args.normalize.std) == 4
