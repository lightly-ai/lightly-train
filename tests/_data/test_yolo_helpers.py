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


def test_oriented_bbox_from_corners__axis_aligned() -> None:
    """Test conversion of an axis-aligned rectangle."""
    # Rectangle from (0.25, 0.25) to (0.75, 0.75) - a square
    corners = np.array([[0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]])
    result = yolo_helpers.oriented_bbox_from_corners(corners)

    # Centroid should be at (0.5, 0.5)
    assert np.isclose(result[0, 0], 0.5)
    assert np.isclose(result[0, 1], 0.5)
    # Width and height should be 0.5
    assert np.isclose(result[0, 2], 0.5)
    assert np.isclose(result[0, 3], 0.5)
    # Angle should be approximately 0 for axis-aligned
    angle = abs(result[0, 4])
    assert np.isclose(angle, 0.0, atol=1e-6)


def test_oriented_bbox_from_corners__rotated() -> None:
    """Test conversion of a rotated rectangle."""
    # A 45-degree rotated square
    corners = np.array(
        [
            [
                0.5,
                0.3,  # top
                0.7,
                0.5,  # right
                0.5,
                0.7,  # bottom
                0.3,
                0.5,
            ]  # left
        ]
    )
    result = yolo_helpers.oriented_bbox_from_corners(corners)

    # asserting is 45deg
    x1, y1, x2, y2 = corners[0, [0, 1, 2, 3]]
    delta_x = x2 - x1
    delta_y = y2 - y1
    assert np.isclose(delta_x, delta_y)

    # Centroid should be at (0.5, 0.5)
    assert np.isclose(result[0, 0], 0.5)
    assert np.isclose(result[0, 1], 0.5)
    # Width and height should be equal for a rotated square
    # (the diagonal of the square)
    assert result[0, 2] > 0
    assert result[0, 3] > 0


def test_oriented_bbox_from_corners__multiple_boxes() -> None:
    """Test conversion of multiple boxes."""
    # Two axis-aligned boxes
    corners = np.array(
        [
            [0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75],  # Box 1
            [0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9],  # Box 2
        ]
    )
    result = yolo_helpers.oriented_bbox_from_corners(corners)

    assert result.shape == (2, 5)
    # First box: centroid at (0.5, 0.5), size 0.5x0.5
    assert np.isclose(result[0, 0], 0.5)
    assert np.isclose(result[0, 1], 0.5)
    # Second box: centroid at (0.75, 0.75), size 0.3x0.3
    assert np.isclose(result[1, 0], 0.75)
    assert np.isclose(result[1, 1], 0.75)


def test_oriented_bbox_from_corners__empty() -> None:
    """Test with empty input."""
    corners = np.zeros((0, 8), dtype=np.float64)
    result = yolo_helpers.oriented_bbox_from_corners(corners)

    assert result.shape == (0, 5)
    assert result.dtype == np.float64
