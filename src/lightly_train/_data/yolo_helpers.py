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

from lightly_train.types import NDArrayBinaryMask, NDArrayBinaryMasks, NDArrayPolygon


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
