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


    def test_no_category_with_keypoints_and_no_user_spec(self) -> None:
        with pytest.raises(ValueError, match="Pass it explicitly"):
            keypoint_helpers.resolve_coco_keypoint_set(
                categories=[_category(1, name="ball")],
                included_class_ids=[1],
                keypoints=None,
            )

    def test_no_category_with_keypoints_uses_user_spec(self) -> None:
        keypoints = KeypointSetArgs(num_keypoints=2, sigmas=[0.1, 0.2])
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[_category(1, name="ball")],
            included_class_ids=[1],
            keypoints=keypoints,
        )
        assert keypoint_set == keypoints

    def test_user_spec_supplies_sigmas_and_flip_idx(self) -> None:
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[_category(1, keypoints=["a", "b"])],
            included_class_ids=[1],
            keypoints=KeypointSetArgs(
                num_keypoints=2, sigmas=[0.1, 0.2], flip_idx=[1, 0]
            ),
        )
        assert keypoint_set.names == ["a", "b"]
        assert keypoint_set.sigmas == [0.1, 0.2]
        assert keypoint_set.flip_idx == [1, 0]

    def test_conflicting_num_keypoints(self) -> None:
        with pytest.raises(ValueError, match="Conflicting values for 'num_keypoints'"):
            keypoint_helpers.resolve_coco_keypoint_set(
                categories=[_category(1, keypoints=["a", "b"])],
                included_class_ids=[1],
                keypoints=KeypointSetArgs(num_keypoints=3),
            )

    def test_conflicting_names(self) -> None:
        with pytest.raises(ValueError, match="Conflicting values for 'names'"):
            keypoint_helpers.resolve_coco_keypoint_set(
                categories=[_category(1, keypoints=["a", "b"])],
                included_class_ids=[1],
                keypoints=KeypointSetArgs(num_keypoints=2, names=["x", "y"]),
            )


class TestBboxFromKeypoints:
    def test_tight_box_over_labeled_keypoints(self) -> None:
        bbox = keypoint_helpers.bbox_from_keypoints(
            keypoints_xy=[[0.2, 0.4], [0.6, 0.8], [0.0, 0.0]],
            visibility=[2, 1, 0],
        )
        # The unlabeled keypoint at the origin must not stretch the box.
        assert bbox == pytest.approx([0.4, 0.6, 0.4, 0.4])

    def test_none_when_nothing_is_labeled(self) -> None:
        bbox = keypoint_helpers.bbox_from_keypoints(
            keypoints_xy=[[0.0, 0.0], [0.0, 0.0]], visibility=[0, 0]
        )
        assert bbox is None

    def test_single_labeled_keypoint_gives_zero_size_box(self) -> None:
        bbox = keypoint_helpers.bbox_from_keypoints(
            keypoints_xy=[[0.3, 0.7], [0.0, 0.0]], visibility=[2, 0]
        )
        assert bbox == pytest.approx([0.3, 0.7, 0.0, 0.0])

    def test_occluded_keypoints_count_as_labeled(self) -> None:
        bbox = keypoint_helpers.bbox_from_keypoints(
            keypoints_xy=[[0.2, 0.2], [0.4, 0.4]], visibility=[1, 1]
        )
        assert bbox == pytest.approx([0.3, 0.3, 0.2, 0.2])
