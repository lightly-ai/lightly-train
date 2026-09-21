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
from pydantic import ValidationError

from lightly_train._data import keypoint_helpers
from lightly_train._data.keypoint_helpers import KeypointSetArgs


class TestKeypointSetArgs:
    def test_minimal(self) -> None:
        keypoint_set = KeypointSetArgs(num_keypoints=3)
        assert keypoint_set.num_keypoints == 3
        assert keypoint_set.names is None
        assert keypoint_set.sigmas is None
        assert keypoint_set.flip_idx is None
        assert keypoint_set.skeleton is None

    def test_full(self) -> None:
        keypoint_set = KeypointSetArgs(
            num_keypoints=3,
            names=["a", "b", "c"],
            sigmas=[0.1, 0.2, 0.3],
            flip_idx=[0, 2, 1],
            skeleton=[[0, 1], [1, 2]],
        )
        assert keypoint_set.names == ["a", "b", "c"]
        assert keypoint_set.sigmas == [0.1, 0.2, 0.3]
        assert keypoint_set.flip_idx == [0, 2, 1]
        assert keypoint_set.skeleton == [[0, 1], [1, 2]]

    def test_num_keypoints_zero(self) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            KeypointSetArgs(num_keypoints=0)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("names", ["a", "b"]),
            ("sigmas", [0.1, 0.2]),
            ("flip_idx", [0, 1]),
        ],
    )
    def test_length_mismatch(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError, match=f"'{field}' to have 3 entries"):
            KeypointSetArgs(num_keypoints=3, **{field: value})

    @pytest.mark.parametrize("sigmas", [[0.1, 0.0, 0.3], [0.1, -0.2, 0.3]])
    def test_sigmas_not_positive(self, sigmas: List[float]) -> None:
        with pytest.raises(ValidationError, match="'sigmas' to be positive"):
            KeypointSetArgs(num_keypoints=3, sigmas=sigmas)

    def test_sigmas_accepts_int(self) -> None:
        # A YAML list such as `sigmas: [1, 0.5, 0.5]` gives an int for the first entry.
        keypoint_set = KeypointSetArgs(num_keypoints=3, sigmas=[1, 0.5, 0.5])
        assert keypoint_set.sigmas == [1.0, 0.5, 0.5]

    @pytest.mark.parametrize(
        "flip_idx",
        [
            [0, 1, 3],  # out of range
            [0, 0, 2],  # duplicates one keypoint and drops another
        ],
    )
    def test_flip_idx_not_a_permutation(self, flip_idx: List[int]) -> None:
        with pytest.raises(ValidationError, match="'flip_idx' to be a permutation"):
            KeypointSetArgs(num_keypoints=3, flip_idx=flip_idx)

    def test_flip_idx_not_involutive(self) -> None:
        # A 3-cycle is a permutation, but flipping twice would not restore the order.
        with pytest.raises(ValidationError, match="'flip_idx' to be its own inverse"):
            KeypointSetArgs(num_keypoints=3, flip_idx=[1, 2, 0])

    def test_flip_idx_identity_is_valid(self) -> None:
        assert KeypointSetArgs(num_keypoints=3, flip_idx=[0, 1, 2]).flip_idx == [
            0,
            1,
            2,
        ]

    def test_skeleton_pair_length(self) -> None:
        with pytest.raises(ValidationError, match="connect exactly two keypoints"):
            KeypointSetArgs(num_keypoints=3, skeleton=[[0, 1, 2]])

    @pytest.mark.parametrize("pair", [[0, 3], [-1, 1]])
    def test_skeleton_index_out_of_range(self, pair: List[int]) -> None:
        with pytest.raises(
            ValidationError, match=r"'skeleton' index to be in \[0, 2\]"
        ):
            KeypointSetArgs(num_keypoints=3, skeleton=[pair])

    def test_skeleton_self_loop(self) -> None:
        with pytest.raises(ValidationError, match="two different keypoints"):
            KeypointSetArgs(num_keypoints=3, skeleton=[[1, 1]])

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            KeypointSetArgs(num_keypoints=3, kpt_shape=[3, 3])  # type: ignore[call-arg]


class TestValidateKptShape:
    def test_valid(self) -> None:
        assert keypoint_helpers.validate_kpt_shape([17, 3]) == (17, 3)
        assert keypoint_helpers.validate_kpt_shape([12, 2]) == (12, 2)

    @pytest.mark.parametrize("kpt_shape", [[17], [17, 3, 1], []])
    def test_wrong_length(self, kpt_shape: List[int]) -> None:
        with pytest.raises(ValueError, match="'kpt_shape' to have two entries"):
            keypoint_helpers.validate_kpt_shape(kpt_shape)

    def test_zero_keypoints(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            keypoint_helpers.validate_kpt_shape([0, 3])

    @pytest.mark.parametrize("num_dims", [1, 4])
    def test_bad_num_dims(self, num_dims: int) -> None:
        with pytest.raises(ValueError, match="to be 2 for"):
            keypoint_helpers.validate_kpt_shape([17, num_dims])


class TestResolveYOLOKeypointSet:
    def test_kpt_shape_only(self) -> None:
        keypoint_set = keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=[3, 3],
            flip_idx=None,
            kpt_names=None,
            keypoints=None,
            included_class_ids=[0],
        )
        assert keypoint_set.num_keypoints == 3
        # The YOLO format carries neither sigmas nor a skeleton, and nothing is invented.
        assert keypoint_set.sigmas is None
        assert keypoint_set.skeleton is None
        assert keypoint_set.names is None

    def test_flip_idx_and_names(self) -> None:
        keypoint_set = keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=[3, 3],
            flip_idx=[0, 2, 1],
            kpt_names={0: ["a", "b", "c"]},
            keypoints=None,
            included_class_ids=[0],
        )
        assert keypoint_set.flip_idx == [0, 2, 1]
        assert keypoint_set.names == ["a", "b", "c"]

    def test_kpt_names_agreeing_across_classes(self) -> None:
        keypoint_set = keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=[3, 3],
            flip_idx=None,
            kpt_names={0: ["a", "b", "c"], 1: ["a", "b", "c"]},
            keypoints=None,
            included_class_ids=[0, 1],
        )
        assert keypoint_set.names == ["a", "b", "c"]

    def test_kpt_names_disagreeing_across_classes(self) -> None:
        with pytest.raises(ValueError, match="same keypoint names"):
            keypoint_helpers.resolve_yolo_keypoint_set(
                kpt_shape=[3, 3],
                flip_idx=None,
                kpt_names={0: ["a", "b", "c"], 1: ["x", "y", "z"]},
                keypoints=None,
                included_class_ids=[0, 1],
            )

    def test_kpt_names_disagreeing_on_ignored_class(self) -> None:
        # The differing class is not included, so there is no conflict.
        keypoint_set = keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=[3, 3],
            flip_idx=None,
            kpt_names={0: ["a", "b", "c"], 1: ["x", "y", "z"]},
            keypoints=None,
            included_class_ids=[0],
        )
        assert keypoint_set.names == ["a", "b", "c"]

    def test_keypoints_supplies_sigmas_and_skeleton(self) -> None:
        keypoint_set = keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=[3, 3],
            flip_idx=None,
            kpt_names=None,
            keypoints=KeypointSetArgs(
                num_keypoints=3, sigmas=[0.1, 0.2, 0.3], skeleton=[[0, 1]]
            ),
            included_class_ids=[0],
        )
        assert keypoint_set.sigmas == [0.1, 0.2, 0.3]
        assert keypoint_set.skeleton == [[0, 1]]

    def test_conflicting_flip_idx(self) -> None:
        with pytest.raises(ValueError, match="Conflicting values for 'flip_idx'"):
            keypoint_helpers.resolve_yolo_keypoint_set(
                kpt_shape=[3, 3],
                flip_idx=[0, 2, 1],
                kpt_names=None,
                keypoints=KeypointSetArgs(num_keypoints=3, flip_idx=[0, 1, 2]),
                included_class_ids=[0],
            )

    def test_identical_flip_idx_in_both_places(self) -> None:
        keypoint_set = keypoint_helpers.resolve_yolo_keypoint_set(
            kpt_shape=[3, 3],
            flip_idx=[0, 2, 1],
            kpt_names=None,
            keypoints=KeypointSetArgs(num_keypoints=3, flip_idx=[0, 2, 1]),
            included_class_ids=[0],
        )
        assert keypoint_set.flip_idx == [0, 2, 1]

    def test_conflicting_num_keypoints(self) -> None:
        with pytest.raises(ValueError, match="Conflicting values for 'num_keypoints'"):
            keypoint_helpers.resolve_yolo_keypoint_set(
                kpt_shape=[3, 3],
                flip_idx=None,
                kpt_names=None,
                keypoints=KeypointSetArgs(num_keypoints=4),
                included_class_ids=[0],
            )


def _category(
    category_id: int,
    name: str = "person",
    keypoints: List[str] | None = None,
    skeleton: List[List[int]] | None = None,
) -> Dict[str, Any]:
    category: Dict[str, Any] = {"id": category_id, "name": name}
    if keypoints is not None:
        category["keypoints"] = keypoints
    if skeleton is not None:
        category["skeleton"] = skeleton
    return category


class TestResolveCOCOKeypointSet:
    def test_single_category(self) -> None:
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[_category(1, keypoints=["a", "b", "c"])],
            included_class_ids=[1],
            keypoints=None,
        )
        assert keypoint_set.num_keypoints == 3
        assert keypoint_set.names == ["a", "b", "c"]
        # The COCO format carries neither sigmas nor flip pairs.
        assert keypoint_set.sigmas is None
        assert keypoint_set.flip_idx is None

    def test_skeleton_converted_to_zero_based(self) -> None:
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[
                _category(1, keypoints=["a", "b", "c"], skeleton=[[1, 2], [2, 3]])
            ],
            included_class_ids=[1],
            keypoints=None,
        )
        assert keypoint_set.skeleton == [[0, 1], [1, 2]]

    @pytest.mark.parametrize("pair", [[0, 1], [1, 4]])
    def test_skeleton_index_out_of_range(self, pair: List[int]) -> None:
        with pytest.raises(ValueError, match=r"to be in \[1, 3\]"):
            keypoint_helpers.resolve_coco_keypoint_set(
                categories=[_category(1, keypoints=["a", "b", "c"], skeleton=[pair])],
                included_class_ids=[1],
                keypoints=None,
            )

    def test_skeleton_pair_length(self) -> None:
        with pytest.raises(ValueError, match="connect exactly two keypoints"):
            keypoint_helpers.resolve_coco_keypoint_set(
                categories=[
                    _category(1, keypoints=["a", "b", "c"], skeleton=[[1, 2, 3]])
                ],
                included_class_ids=[1],
                keypoints=None,
            )

    def test_two_categories_agreeing(self) -> None:
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[
                _category(1, name="person", keypoints=["a", "b"]),
                _category(2, name="cat", keypoints=["a", "b"]),
            ],
            included_class_ids=[1, 2],
            keypoints=None,
        )
        assert keypoint_set.names == ["a", "b"]

    def test_two_categories_disagreeing(self) -> None:
        with pytest.raises(ValueError, match="same keypoints"):
            keypoint_helpers.resolve_coco_keypoint_set(
                categories=[
                    _category(1, name="person", keypoints=["a", "b"]),
                    _category(2, name="cat", keypoints=["x", "y", "z"]),
                ],
                included_class_ids=[1, 2],
                keypoints=None,
            )

    def test_two_categories_disagreeing_one_ignored(self) -> None:
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[
                _category(1, name="person", keypoints=["a", "b"]),
                _category(2, name="cat", keypoints=["x", "y", "z"]),
            ],
            included_class_ids=[1],
            keypoints=None,
        )
        assert keypoint_set.names == ["a", "b"]

    def test_category_without_keypoints_is_ignored(self) -> None:
        keypoint_set = keypoint_helpers.resolve_coco_keypoint_set(
            categories=[
                _category(1, name="person", keypoints=["a", "b"]),
                _category(2, name="ball"),
            ],
            included_class_ids=[1, 2],
            keypoints=None,
        )
        assert keypoint_set.names == ["a", "b"]

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
