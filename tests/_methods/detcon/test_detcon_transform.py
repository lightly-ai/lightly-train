#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from torch import Tensor

from lightly_train._methods.detcon.detcon_transform import (
    DetConBTransform,
    DetConSTransform,
)
from lightly_train.types import TransformInput


def test_detcons_transform_shapes() -> None:
    img_np = np.random.uniform(0, 255, size=(1234, 1234, 3))
    mask_np = np.random.uniform(0, 16, size=(1234, 1234, 1))
    input: TransformInput = {
        "image": img_np.astype(np.uint8),
        "mask": mask_np.astype(np.uint8),
    }

    transform_args = DetConSTransform.transform_args_cls()()
    transform = DetConSTransform(transform_args)

    transformed = transform(input)
    assert len(transformed) == 2
    image1 = transformed[0]["image"]
    image2 = transformed[1]["image"]
    mask1 = transformed[0]["mask"]
    mask2 = transformed[1]["mask"]
    assert isinstance(image1, Tensor)
    assert isinstance(image2, Tensor)
    assert image1.shape == (3, 224, 224)
    assert image2.shape == (3, 224, 224)
    assert isinstance(mask1, Tensor)
    assert isinstance(mask2, Tensor)
    assert mask1.shape == (224, 224, 1)
    assert mask2.shape == (224, 224, 1)


def test_detconb_transform_shapes() -> None:
    img_np = np.random.uniform(0, 255, size=(1234, 1234, 3))
    mask_np = np.random.uniform(0, 16, size=(1234, 1234, 1))
    input: TransformInput = {
        "image": img_np.astype(np.uint8),
        "mask": mask_np.astype(np.uint8),
    }

    transform_args = DetConBTransform.transform_args_cls()()
    transform = DetConBTransform(transform_args)

    transformed = transform(input)
    assert len(transformed) == 2
    image1 = transformed[0]["image"]
    image2 = transformed[1]["image"]
    mask1 = transformed[0]["mask"]
    mask2 = transformed[1]["mask"]
    assert isinstance(image1, Tensor)
    assert isinstance(image2, Tensor)
    assert image1.shape == (3, 224, 224)
    assert image2.shape == (3, 224, 224)
    assert isinstance(mask1, Tensor)
    assert isinstance(mask2, Tensor)
    assert mask1.shape == (224, 224, 1)
    assert mask2.shape == (224, 224, 1)
