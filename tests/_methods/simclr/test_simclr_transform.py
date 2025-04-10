#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np

from lightly_train._methods.simclr.simclr_transform import (
    SimCLRTransform,
)
from lightly_train.types import NDArrayImage, TransformInput


def test_simclr_transform_shapes() -> None:
    img_np: NDArrayImage = np.random.uniform(0, 255, size=(1234, 1234, 3)).astype(
        np.uint8
    )
    input: TransformInput = {"image": img_np}

    transform_args = SimCLRTransform.transform_args_cls()()
    transform = SimCLRTransform(transform_args)

    views = transform(input)
    assert len(views) == 2
    for view in views:
        assert view["image"].shape == (3, 224, 224)
