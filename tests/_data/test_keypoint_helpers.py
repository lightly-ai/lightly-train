#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from lightly_train._data import keypoint_helpers


@pytest.mark.parametrize("visibility", [0, 1, 2, 2.0, 0.0])
def test_parse_visibility__valid(visibility: float) -> None:
    assert keypoint_helpers.parse_visibility(visibility) == int(visibility)


@pytest.mark.parametrize("visibility", [1.9, 0.5, -1, 3, 2.5])
def test_parse_visibility__invalid(visibility: float) -> None:
    assert keypoint_helpers.parse_visibility(visibility) is None


def test_validate_kpt_shape__valid() -> None:
    assert keypoint_helpers.validate_kpt_shape((17, 3)) == (17, 3)
    assert keypoint_helpers.validate_kpt_shape((12, 2)) == (12, 2)


def test_validate_kpt_shape__zero_keypoints() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        keypoint_helpers.validate_kpt_shape((0, 3))


@pytest.mark.parametrize("num_dims", [1, 4])
def test_validate_kpt_shape__invalid_num_dims(num_dims: int) -> None:
    with pytest.raises(ValueError, match="to be 2 for"):
        keypoint_helpers.validate_kpt_shape((17, num_dims))


def test_validate_flip_idx__valid() -> None:
    keypoint_helpers.validate_flip_idx([0, 2, 1], 3)


@pytest.mark.parametrize("flip_idx", [[0, 0, 2], [1, 2, 0]])
def test_validate_flip_idx__invalid(flip_idx: List[int]) -> None:
    with pytest.raises(ValueError, match="flip_idx"):
        keypoint_helpers.validate_flip_idx(flip_idx, 3)


def test_validate_kpt_names__valid() -> None:
    keypoint_helpers.validate_kpt_names({0: ["a", "b", "c"]}, 3)


def test_validate_kpt_names__invalid() -> None:
    with pytest.raises(ValueError, match="kpt_names"):
        keypoint_helpers.validate_kpt_names({0: ["a", "b"]}, 3)


def test_validate_kpt_oks_sigmas__valid() -> None:
    keypoint_helpers.validate_kpt_oks_sigmas([0.1, 0.2, 0.3], 3)


@pytest.mark.parametrize("sigmas", [[0.1, 0.2], [0.1, 0.0, 0.3]])
def test_validate_kpt_oks_sigmas__invalid(sigmas: List[float]) -> None:
    with pytest.raises(ValueError, match="kpt_oks_sigmas"):
        keypoint_helpers.validate_kpt_oks_sigmas(sigmas, 3)


def test_get_coco_num_keypoints__shared_keypoints() -> None:
    categories: List[Dict[str, Any]] = [
        {"id": 1, "keypoints": ["a", "b"]},
        {"id": 2, "keypoints": ["a", "b"]},
    ]
    assert keypoint_helpers.get_coco_num_keypoints(categories, [1, 2]) == 2


def test_get_coco_num_keypoints__ignored_category_with_different_keypoints() -> None:
    categories: List[Dict[str, Any]] = [
        {"id": 1, "keypoints": ["a", "b"]},
        {"id": 2, "keypoints": ["a", "b", "c"]},
    ]
    assert keypoint_helpers.get_coco_num_keypoints(categories, [1]) == 2


def test_get_coco_num_keypoints__included_categories_with_different_keypoints() -> None:
    categories: List[Dict[str, Any]] = [
        {"id": 1, "keypoints": ["a", "b"]},
        {"id": 2, "keypoints": ["a", "b", "c"]},
    ]
    with pytest.raises(ValueError, match="same keypoints"):
        keypoint_helpers.get_coco_num_keypoints(categories, [1, 2])


def test_bbox_from_keypoints__labeled_keypoints() -> None:
    bbox = keypoint_helpers.bbox_from_keypoints(
        keypoints_xy=[[0.2, 0.4], [0.6, 0.8], [0.0, 0.0]],
        visibility=[2, 1, 0],
    )
    assert bbox == pytest.approx([0.4, 0.6, 0.4, 0.4])


def test_bbox_from_keypoints__no_labeled_keypoints() -> None:
    assert (
        keypoint_helpers.bbox_from_keypoints(
            keypoints_xy=[[0.0, 0.0], [0.0, 0.0]], visibility=[0, 0]
        )
        is None
    )
