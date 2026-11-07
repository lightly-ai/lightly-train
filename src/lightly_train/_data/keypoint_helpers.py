#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from collections.abc import Iterable
from enum import IntEnum
from typing import Any


class Visibility(IntEnum):
    """Keypoint visibility flag, shared by COCO and YOLO pose."""

    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2


def parse_visibility(visibility: Any) -> Visibility | None:
    """Returns the visibility flag, or None if the value is not a valid flag.

    Exporters commonly write the flag as a float, e.g. ``2.000000``. A fractional
    value such as ``1.9`` is not a flag and must not be truncated to one.
    """
    if isinstance(visibility, bool) or not isinstance(visibility, (int, float)):
        return None
    if not math.isfinite(visibility):
        return None
    flag = int(visibility)
    if flag != visibility:
        return None
    try:
        return Visibility(flag)
    except ValueError:
        return None


def validate_kpt_shape(kpt_shape: tuple[int, int]) -> tuple[int, int]:
    """Validates a YOLO ``kpt_shape``.

    - ``kpt_shape`` is ``(num_keypoints, num_dims)``.
    - ``num_dims`` is 2 for (x, y) or 3 for (x, y, visibility).
    - Example: ``(17, 3)`` for COCO-style human pose.
    """
    num_keypoints, num_dims = kpt_shape
    if num_keypoints < 1:
        raise ValueError(
            f"Expected the number of keypoints in 'kpt_shape' to be at least 1, got "
            f"{num_keypoints}."
        )
    if num_dims not in (2, 3):
        raise ValueError(
            f"Expected the number of dimensions in 'kpt_shape' to be 2 for (x, y) or "
            f"3 for (x, y, visibility), got {num_dims}."
        )
    return kpt_shape


def validate_flip_idx(flip_idx: list[int] | None, num_keypoints: int) -> None:
    """Validates a YOLO keypoint flip mapping.

    - ``flip_idx[i]`` is the index keypoint ``i`` takes after a horizontal flip.
    - Left/right pairs swap, center keypoints map to themselves. Example for
      ``[nose, left_eye, right_eye]``: ``[0, 2, 1]``.
    - Must be a permutation that is its own inverse, since flipping twice is the
      identity.
    - ``None`` means no mapping; horizontal flips must then be disabled.
    """
    if flip_idx is None:
        return
    if len(flip_idx) != num_keypoints:
        raise ValueError(
            f"Expected 'flip_idx' to have {num_keypoints} entries, got {len(flip_idx)}."
        )
    if sorted(flip_idx) != list(range(num_keypoints)):
        raise ValueError(
            f"Expected 'flip_idx' to be a permutation of [0, {num_keypoints - 1}], "
            f"got {flip_idx}."
        )
    if any(flip_idx[flip_idx[i]] != i for i in range(num_keypoints)):
        raise ValueError("Expected 'flip_idx' to be its own inverse.")


def validate_kpt_names(
    kpt_names: dict[int, list[str]] | None, num_keypoints: int
) -> None:
    """Validates optional YOLO keypoint names.

    - Maps class ID to one name per keypoint, e.g. ``{0: ["nose", "left_eye"]}``.
    - Every list must have ``num_keypoints`` entries.
    """
    if kpt_names is None:
        return
    for class_id, names in kpt_names.items():
        if len(names) != num_keypoints:
            raise ValueError(
                f"Expected 'kpt_names' for class {class_id} to have {num_keypoints} "
                f"entries, got {len(names)}."
            )


def validate_kpt_oks_sigmas(
    kpt_oks_sigmas: list[float] | None, num_keypoints: int
) -> None:
    """Validates optional YOLO OKS sigmas.

    - One positive sigma per keypoint, used by the OKS (object keypoint similarity)
      metric.
    - Sigma is the keypoint's localization tolerance: larger is more forgiving.
      In COCO-17, eyes use 0.025 and hips 0.107.
    """
    if kpt_oks_sigmas is None:
        return
    if len(kpt_oks_sigmas) != num_keypoints or any(
        sigma <= 0 for sigma in kpt_oks_sigmas
    ):
        raise ValueError(
            f"Expected 'kpt_oks_sigmas' to be {num_keypoints} positive values, got "
            f"{kpt_oks_sigmas}."
        )


def get_coco_num_keypoints(
    categories: list[dict[str, Any]], included_class_ids: Iterable[int]
) -> int:
    """Returns the shared number of COCO keypoints.

    - Read from ``categories[].keypoints`` of the included categories.
    - All included categories that declare keypoints must declare the same ones.
    - Ignored categories may differ.
    """
    included = set(included_class_ids)
    declared = {
        category["id"]: category["keypoints"]
        for category in categories
        if category["id"] in included and category.get("keypoints")
    }
    distinct = {tuple(names) for names in declared.values()}
    if len(distinct) > 1:
        details = ", ".join(
            f"category {category_id}: {names}"
            for category_id, names in sorted(declared.items())
        )
        raise ValueError(
            "Expected all included categories to declare the same keypoints, but got "
            f"{details}."
        )
    if not distinct:
        raise ValueError(
            "No included category in the annotations declares a 'keypoints' field."
        )
    return len(next(iter(distinct)))


def bbox_from_keypoints(
    keypoints_xy: list[list[float]], visibility: list[int]
) -> list[float] | None:
    """Returns the tight bounding box around labeled keypoints."""
    labeled = [
        point
        for point, vis in zip(keypoints_xy, visibility)
        if vis != Visibility.UNLABELED
    ]
    if not labeled:
        return None
    xs = [point[0] for point in labeled]
    ys = [point[1] for point in labeled]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None
    return [
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        width,
        height,
    ]


def resolve_bbox(
    bbox: list[float] | None,
    keypoints_xy: list[list[float]],
    visibility: list[int],
) -> list[float] | None:
    """Returns the bbox if it has an area, else the box around the labeled keypoints.

    Bbox is (x_center, y_center, width, height). Returns None if neither has an area.
    """
    if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
        return bbox
    return bbox_from_keypoints(keypoints_xy=keypoints_xy, visibility=visibility)
