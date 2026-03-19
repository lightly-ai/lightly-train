#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from typing_extensions import Literal

from lightly_train.types import (
    NDArray4Corners,
    NDArrayBinaryMask,
    NDArrayBinaryMasks,
    NDArrayOBBoxes,
    NDArrayPolygon,
)


def get_image_and_labels_dirs(
    path: Path,
    train: Path,
    val: Path,
    test: Path | None,
    mode: Literal["train", "val", "test"],
) -> tuple[Path | None, Path | None]:
    train_img_dir = path / train
    val_img_dir = path / val
    test_img_dir = path / test if test else None

    def _replace_first_images_with_labels(path: Path) -> Path:
        """Replaces only the first occurrence of 'images' with 'labels' in a Path."""
        parts = list(path.parts)
        for i, part in enumerate(parts):
            if part == "images":
                parts[i] = "labels"
                break
        return Path(*parts)

    train_label_path = _replace_first_images_with_labels(train)
    val_label_path = _replace_first_images_with_labels(val)
    test_label_path = _replace_first_images_with_labels(test) if test else None

    train_label_dir = path / train_label_path
    val_label_dir = path / val_label_path
    test_label_dir = path / test_label_path if test_label_path else None

    if mode == "train":
        return train_img_dir, train_label_dir
    elif mode == "val":
        return val_img_dir, val_label_dir
    elif mode == "test":
        return test_img_dir, test_label_dir
    else:
        raise ValueError(f"Unknown mode: {mode}")


def binary_mask_from_polygon(
    polygon: NDArrayPolygon, height: int, width: int
) -> NDArrayBinaryMask:
    """Convert a YOLO polygon to a binary mask.

    Args:
        polygon:
            Numpy array of shape (n_points*2,) containing the polygon points
            in normalized coordinates [0, 1].
        height:
            Height of the image.
        width:
            Width of the image.

    Returns:
        Binary mask with shape (H, W) where all points inside the polygon are 1.
    """
    mask = Image.new("1", (width, height), 0)
    points = [(x * width, y * height) for x, y in zip(polygon[0::2], polygon[1::2])]

    # Split the points into multiple polygons if necessary. A single YOLO polygon line
    # can contain multiple polygons connected by linking the last point of one polygon
    # to the first point of the next polygon with a line.
    # See https://github.com/ultralytics/yolov5/issues/11476#issuecomment-1537281864
    # for details.
    polygons = []
    current_polygon: list[tuple[float, float]] = []
    prev_point = None
    for point in points:
        if prev_point == point:
            # Skip duplicate points
            continue
        if len(current_polygon) >= 3 and current_polygon[0] == point:
            # Polygon is closed if current point matches the first point.
            # Start a new polygon.
            current_polygon.append(point)
            if len(current_polygon) >= 4:
                polygons.append(current_polygon)
            current_polygon = []
        else:
            current_polygon.append(point)
        prev_point = point

    # Add last polygon. Only 3 points required as it might not be closed.
    if len(current_polygon) >= 3:
        polygons.append(current_polygon)

    for poly in polygons:
        ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
    return np.array(mask, dtype=np.bool_)


def binary_masks_from_polygons(
    polygons: list[NDArrayPolygon], height: int, width: int
) -> NDArrayBinaryMasks:
    """Convert a list of YOLO polygons to a stack of binary masks.

    Args:
        polygons:
            List of numpy arrays, each of shape (n_points*2,) containing the
            polygon points in normalized coordinates [0, 1].
        height:
            Height of the image.
        width:
            Width of the image.

    Returns:
        Stack of binary masks with shape (n_polygons, H, W) where all points
        inside each polygon are 1.
    """
    binary_masks = [
        binary_mask_from_polygon(polygon, height, width) for polygon in polygons
    ]
    if binary_masks:
        return np.stack(binary_masks)
    else:
        return np.zeros((0, height, width), dtype=np.bool_)


def oriented_bbox_from_corners(corners: NDArray4Corners) -> NDArrayOBBoxes:
    """Convert 4-corner format to (cx, cy, w, h, angle) format.

    Uses the mmrotate geometric approach: the center is the midpoint of opposite
    corners, width/height are the longer/shorter edge lengths, and the angle is
    derived from the longer edge direction.

    Args:
        corners:
            Array of shape (n_boxes, 8) with coordinates
            (x0, y0, x1, y1, x2, y2, x3, y3) in normalized [0, 1] coordinates.
            Corners should be in clockwise or counter-clockwise order.

    Returns:
        Array of shape (n_boxes, 5) with (cx, cy, w, h, angle) in normalized
        coordinates. Angle is in radians, in the range (-pi/2, 0].
    """
    if corners.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)

    # Reshape to (n_boxes, 4, 2) where last dim is (x, y)
    pts = corners.reshape(-1, 4, 2)

    # Center: midpoint of opposite corners (p0 and p2)
    cx = (pts[:, 0, 0] + pts[:, 2, 0]) / 2
    cy = (pts[:, 0, 1] + pts[:, 2, 1]) / 2

    # Edge lengths from p0 to its two adjacent corners (p1 and p3)
    edge1_sq = (pts[:, 1, 0] - pts[:, 0, 0]) ** 2 + (pts[:, 1, 1] - pts[:, 0, 1]) ** 2
    edge2_sq = (pts[:, 3, 0] - pts[:, 0, 0]) ** 2 + (pts[:, 3, 1] - pts[:, 0, 1]) ** 2
    edge1_len = np.sqrt(edge1_sq)
    edge2_len = np.sqrt(edge2_sq)

    # Width is the longer edge, height is the shorter
    long_edge = np.maximum(edge1_len, edge2_len)
    short_edge = np.minimum(edge1_len, edge2_len)

    # Angle of each edge
    edge1_dx = pts[:, 1, 0] - pts[:, 0, 0]
    edge1_dy = pts[:, 1, 1] - pts[:, 0, 1]
    edge2_dx = pts[:, 3, 0] - pts[:, 0, 0]
    edge2_dy = pts[:, 3, 1] - pts[:, 0, 1]

    angle1 = np.arctan2(edge1_dy, edge1_dx)
    angle2 = np.arctan2(edge2_dy, edge2_dx)

    # Use the longer edge's angle
    use_edge1 = edge1_len >= edge2_len
    angle = np.where(use_edge1, angle1, angle2)

    # Normalize to (-pi/2, 0]: if angle > 0, subtract pi/2 and swap w/h.
    # After this, width >= height always holds.
    needs_swap = angle > 0
    angle = np.where(needs_swap, angle - np.pi / 2, angle)
    width = np.where(needs_swap, short_edge, long_edge)
    height = np.where(needs_swap, long_edge, short_edge)

    return np.stack([cx, cy, width, height, angle], axis=1)
