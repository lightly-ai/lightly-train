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


def test_validate_kpt_shape__valid() -> None:
    assert keypoint_helpers.validate_kpt_shape((17, 3)) == (17, 3)


@pytest.mark.parametrize("kpt_shape", [(0, 3), (3, 4)])
def test_validate_kpt_shape__invalid(kpt_shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        keypoint_helpers.validate_kpt_shape(kpt_shape)


def test_validate_flip_idx__valid() -> None:
    keypoint_helpers.validate_flip_idx([0, 2, 1], 3)


@pytest.mark.parametrize("flip_idx", [[0, 0, 2], [1, 2, 0]])
def test_validate_flip_idx__invalid(flip_idx: List[int]) -> None:
    with pytest.raises(ValueError, match="flip_idx"):
        keypoint_helpers.validate_flip_idx(flip_idx, 3)


def test_validate_kpt_names__invalid() -> None:
    with pytest.raises(ValueError, match="kpt_names"):
        keypoint_helpers.validate_kpt_names({0: ["a", "b"]}, 3)


def test_validate_kpt_oks_sigmas__valid() -> None:
    keypoint_helpers.validate_kpt_oks_sigmas([0.1, 0.2, 0.3], 3)


def test_validate_kpt_oks_sigmas__invalid() -> None:
    with pytest.raises(ValueError, match="kpt_oks_sigmas"):
        keypoint_helpers.validate_kpt_oks_sigmas([0.1, 0.0, 0.3], 3)


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
