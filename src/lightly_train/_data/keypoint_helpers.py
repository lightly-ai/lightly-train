#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Visibility convention, shared by COCO and YOLO pose.
VISIBILITY_UNLABELED = 0
VISIBILITY_OCCLUDED = 1
VISIBILITY_VISIBLE = 2
VISIBILITIES = (VISIBILITY_UNLABELED, VISIBILITY_OCCLUDED, VISIBILITY_VISIBLE)


def validate_kpt_shape(kpt_shape: tuple[int, int]) -> tuple[int, int]:
    """Validates a YOLO ``kpt_shape``."""
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
    """Validates a YOLO keypoint flip mapping."""
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
    """Validates optional YOLO keypoint names."""
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
    """Validates optional YOLO OKS sigmas."""
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
    """Returns the shared number of COCO keypoints."""
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
        if vis != VISIBILITY_UNLABELED
    ]
    if not labeled:
        return None
    xs = [point[0] for point in labeled]
    ys = [point[1] for point in labeled]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        x_max - x_min,
        y_max - y_min,
    ]
