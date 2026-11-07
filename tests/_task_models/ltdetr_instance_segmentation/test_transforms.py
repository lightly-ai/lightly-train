#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._task_models.ltdetr_instance_segmentation.transforms import (
    LTDETRInstanceSegmentationTrainTransformArgs,
)
from lightly_train._transforms.transform import NormalizeArgs


@pytest.mark.parametrize("num_channels, enabled", [(3, True), (4, False)])
def test_train_transform_args_resolve_auto__rgb_only_transforms(
    num_channels: int, enabled: bool
) -> None:
    transform_args = LTDETRInstanceSegmentationTrainTransformArgs(
        num_channels=num_channels
    )
    transform_args.resolve_auto(model_init_args={})

    assert (transform_args.photometric_distort is not None) is enabled
    assert (transform_args.mosaic is not None) is enabled
    assert isinstance(transform_args.normalize, NormalizeArgs)
    assert len(transform_args.normalize.mean) == num_channels
