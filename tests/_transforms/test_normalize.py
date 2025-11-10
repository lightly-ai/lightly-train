#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np

from lightly_train._transforms.normalize import NormalizeDtypeAware


def test_standard_normalization_float_image() -> None:
    transform = NormalizeDtypeAware()
    image = np.array(
        [
            [[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]],
            [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
        ],
        dtype=np.float32,
    )

    result = transform(image=image)["image"]

    mean = np.asarray(transform.mean, dtype=np.float32)
    std = np.asarray(transform.std, dtype=np.float32)

    expected = (image.astype(np.float32) - mean) / std
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_standard_normalization_uint8_image() -> None:
    transform = NormalizeDtypeAware()
    image = np.array(
        [
            [[0, 64, 128], [192, 255, 32]],
            [[16, 48, 80], [112, 144, 176]],
        ],
        dtype=np.uint8,
    )

    result = transform(image=image)["image"]

    mean = np.asarray(transform.mean, dtype=np.float32) * float(
        transform.max_pixel_value
    )
    std = np.asarray(transform.std, dtype=np.float32) * float(transform.max_pixel_value)

    expected = (image.astype(np.float32) - mean) / std
    np.testing.assert_allclose(result, expected, atol=1e-6)
