#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np

from lightly_train._data import yolo_helpers


def test_binary_mask_from_polygon() -> None:
    poly = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.1])
    mask = yolo_helpers.binary_mask_from_polygon(polygon=poly, height=10, width=10)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    assert np.all(mask == expected)


def test_binary_mask_from_polygon__multiple() -> None:
    poly = np.array(
        [
            # First polygon
            0.0,
            0.0,
            0.0,
            0.3,
            0.3,
            0.3,
            0.0,
            0.0,
            # Second polygon
            0.5,
            0.5,
            0.5,
            0.8,
            0.8,
            0.8,  # Last poly doesn't need to be closed
        ]
    )
    mask = yolo_helpers.binary_mask_from_polygon(polygon=poly, height=10, width=10)
    print(repr(mask.astype(np.int_)))
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.bool_,
    )
    assert np.all(mask == expected)
