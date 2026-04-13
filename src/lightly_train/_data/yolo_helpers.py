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


def split_yolo_polygon(polygon: list[float]) -> list[list[float]]:
    """Split a flat YOLO polygon coordinate list into individual polygon segments.

    A single YOLO polygon line can encode multiple disconnected polygons by
    closing each sub-polygon with a repeated first point before starting the next.
    See https://github.com/ultralytics/yolov5/issues/11476#issuecomment-1537281864

    Args:
        polygon:
            Flat list of normalized [0, 1] coordinates: [x0, y0, x1, y1, ...].

    Returns:
        List of flat coordinate lists, one per sub-polygon.
    """
    points = list(zip(polygon[0::2], polygon[1::2]))
    segments: list[list[float]] = []
    current: list[tuple[float, float]] = []
    prev: tuple[float, float] | None = None
    for point in points:
        if prev == point:
            # Skip duplicate points.
            continue
        if len(current) >= 3 and current[0] == point:
            # Current point closes the polygon; start a new one.
            current.append(point)
            if len(current) >= 4:
                segments.append([c for pt in current for c in pt])
            current = []
        else:
            current.append(point)
        prev = point
    # Add the last (possibly unclosed) polygon.
    if len(current) >= 3:
        segments.append([c for pt in current for c in pt])
    return segments if segments else [polygon]


def binary_mask_from_polygon(
    polygons: list[NDArrayPolygon], height: int, width: int
) -> NDArrayBinaryMask:
    """Convert a list of polygons for a single instance to a binary mask.

    Args:
        polygons:
            List of numpy arrays, each of shape (n_points*2,) containing the
            polygon points in normalized coordinates [0, 1].
        height:
            Height of the image.
        width:
            Width of the image.

    Returns:
        Binary mask with shape (H, W) where all points inside any polygon are 1.
    """
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = [(x * width, y * height) for x, y in zip(polygon[0::2], polygon[1::2])]
        if len(points) >= 3:
            draw.polygon(points, outline=1, fill=1)
    return np.array(mask, dtype=np.bool_)


def binary_masks_from_polygons(
    polygons: list[list[NDArrayPolygon]], height: int, width: int
) -> NDArrayBinaryMasks:
    """Convert a list of per-instance polygon groups to a stack of binary masks.

    Args:
        polygons:
            List of polygon groups, one per instance. Each group is a list of
            numpy arrays of shape (n_points*2,) in normalized coordinates [0, 1].
        height:
            Height of the image.
        width:
            Width of the image.

    Returns:
        Stack of binary masks with shape (n_instances, H, W).
    """
    binary_masks = [
        binary_mask_from_polygon(polygon_group, height, width)
        for polygon_group in polygons
    ]
    if binary_masks:
        return np.stack(binary_masks)
    else:
        return np.zeros((0, height, width), dtype=np.bool_)


def oriented_bbox_from_corners(corners: NDArray4Corners) -> NDArrayOBBoxes:
    """Convert 4-corner format to (cx, cy, w, h, angle) format.

    Args:
        corners:
            Array of shape (n_boxes, 8) with coordinates
            (x0, y0, x1, y1, x2, y2, x3, y3)
            Corners should be in clockwise or counter-clockwise order.
            Handles the n_boxes == 0 case.

    Returns:
        Array of shape (n_boxes, 5) with (cx, cy, w, h, angle)
        coordinates. Angle is in radians, in the range [-pi/2, pi/2].
    """
    if corners.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float64)

    pts = corners.reshape(-1, 4, 2)

    cx = (pts[:, 0, 0] + pts[:, 2, 0]) / 2
    cy = (pts[:, 0, 1] + pts[:, 2, 1]) / 2

    width = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    height = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    angle = np.arctan2(pts[:, 1, 1] - pts[:, 0, 1], pts[:, 1, 0] - pts[:, 0, 0])

    angle_over = angle > np.pi / 2
    angle_under = angle < -np.pi / 2

    angle[angle_over] -= np.pi
    angle[angle_under] += np.pi

    return np.stack([cx, cy, width, height, angle], axis=1)
