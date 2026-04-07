#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._data.label_helpers import get_class_id_to_internal_class_id_mapping


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
