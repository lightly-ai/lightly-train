#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# * Note: This file is almost identical to tests/_methods/dino/test_dino_transform.py
from __future__ import annotations

import numpy as np

from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
)
from lightly_train.types import NDArrayImage, TransformInput


def test_dinov2_transform_shapes() -> None:
    img_np: NDArrayImage = np.random.uniform(0, 255, size=(1234, 1234, 3)).astype(
        np.uint8
    )
    input: TransformInput = {"image": img_np}

    transform_args = DINOv2ViTTransform.transform_args_cls()()
    transform = DINOv2ViTTransform(transform_args)

    views = transform(input)
    assert len(views) == 2 + 8
    for view in views[:2]:
        assert view["image"].shape == (3, 224, 224)
    for view in views[2:]:
        assert view["image"].shape == (3, 98, 98)
