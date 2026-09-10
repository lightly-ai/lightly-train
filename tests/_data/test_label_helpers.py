#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._data.label_helpers import (
    get_class_id_to_internal_class_id_mapping,
    internal_ordered_class_ids,
)


class TestGetClassIdToInternalClassIdMapping:
    def test__contiguous(self) -> None:
        assert get_class_id_to_internal_class_id_mapping(
            class_ids=[0, 1, 2], ignore_classes=None
        ) == {0: 0, 1: 1, 2: 2}

    def test__non_contiguous(self) -> None:
        assert get_class_id_to_internal_class_id_mapping(
            class_ids=[3, 7, 5], ignore_classes=None
        ) == {3: 0, 7: 1, 5: 2}

    def test__ignore_classes(self) -> None:
        assert get_class_id_to_internal_class_id_mapping(
            class_ids=[0, 1, 2, 3], ignore_classes={1, 3}
        ) == {0: 0, 2: 1}

    def test__ignore_classes_none(self) -> None:
        assert get_class_id_to_internal_class_id_mapping(
            class_ids=[0, 1], ignore_classes=None
        ) == {0: 0, 1: 1}

    def test__ignore_classes_empty(self) -> None:
        assert get_class_id_to_internal_class_id_mapping(
            class_ids=[0, 1], ignore_classes=set()
        ) == {0: 0, 1: 1}

    def test__all_ignored(self) -> None:
        assert (
            get_class_id_to_internal_class_id_mapping(
                class_ids=[0, 1], ignore_classes={0, 1}
            )
            == {}
        )

    def test__empty_class_ids(self) -> None:
        assert (
            get_class_id_to_internal_class_id_mapping(class_ids=[], ignore_classes=None)
            == {}
        )


class TestInternalOrderedClassIds:
    def test__non_contiguous(self) -> None:
        assert internal_ordered_class_ids(
            class_ids=[3, 7, 12], ignore_classes=None
        ) == [3, 7, 12]

    def test__ignore_middle_class(self) -> None:
        # Original: 3->cat, 7->car, 12->dog; ignoring 7 gives internal 0->3, 1->12.
        assert internal_ordered_class_ids(class_ids=[3, 7, 12], ignore_classes={7}) == [
            3,
            12,
        ]

    def test__unsorted_class_ids(self) -> None:
        # No sorting is applied, the order follows the given class ids.
        assert internal_ordered_class_ids(
            class_ids=[12, 3, 7], ignore_classes=None
        ) == [12, 3, 7]

    def test__all_ignored(self) -> None:
        assert internal_ordered_class_ids(class_ids=[0, 1], ignore_classes={0, 1}) == []
