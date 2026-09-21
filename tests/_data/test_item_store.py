#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import MemoryMappedSequence
from lightly_train._data.item_store import ItemStore, ListItemStore, as_item_store

StoreFactory = Callable[[Sequence[Mapping[str, str]], Path], ItemStore]


def _list_store(items: Sequence[Mapping[str, str]], tmp_path: Path) -> ItemStore:
    return ListItemStore(items)


def _memory_mapped_store(
    items: Sequence[Mapping[str, str]], tmp_path: Path
) -> ItemStore:
    mmap_filepath = tmp_path / "test.arrow"
    memory_mapped_sequence.write_items_to_file(items=items, mmap_filepath=mmap_filepath)
    return MemoryMappedSequence[str].from_file(mmap_filepath=mmap_filepath)


# Both implementations must behave identically.
STORE_FACTORIES = [_list_store, _memory_mapped_store]


@pytest.mark.parametrize("make_store", STORE_FACTORIES)
class TestItemStore:
    def test_len_and_getitem(self, make_store: StoreFactory, tmp_path: Path) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": "3"},
            {"image_path": "b.jpg", "class_id": "12"},
        ]
        store = make_store(items, tmp_path)
        assert len(store) == 2
        assert store[0] == items[0]
        assert list(store[0:2]) == items

    def test_value_counts(self, make_store: StoreFactory, tmp_path: Path) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": "3,12"},
            {"image_path": "b.jpg", "class_id": "3,12"},
            {"image_path": "c.jpg", "class_id": "3"},
        ]
        store = make_store(items, tmp_path)
        # Values are counted as they are stored, they are not interpreted.
        assert store.value_counts("class_id") == {"3,12": 2, "3": 1}

    def test_value_counts__whitespace_and_empty_values(
        self, make_store: StoreFactory, tmp_path: Path
    ) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": " 3 "},
            {"image_path": "b.jpg", "class_id": "3"},
            {"image_path": "c.jpg", "class_id": ""},
        ]
        store = make_store(items, tmp_path)
        assert store.value_counts("class_id") == {"3": 2}

    def test_value_counts__unknown_column(
        self, make_store: StoreFactory, tmp_path: Path
    ) -> None:
        store = make_store([{"image_path": "a.jpg"}], tmp_path)
        assert store.value_counts("class_id") == {}

    def test_value_counts__no_items(
        self, make_store: StoreFactory, tmp_path: Path
    ) -> None:
        store = make_store([], tmp_path)
        assert len(store) == 0
        assert store.value_counts("class_id") == {}

    def test_is_item_store(self, make_store: StoreFactory, tmp_path: Path) -> None:
        store = make_store([{"image_path": "a.jpg"}], tmp_path)
        assert isinstance(store, ItemStore)
        # Stores are not wrapped again.
        assert as_item_store(store) is store


class TestAsItemStore:
    def test_wraps_list(self) -> None:
        items = [{"image_path": "a.jpg", "class_id": "3"}]
        store = as_item_store(items)
        assert isinstance(store, ListItemStore)
        assert store[0] == items[0]

    def test_list_is_no_item_store(self) -> None:
        # A plain list cannot count values and must be wrapped.
        assert not isinstance([{"image_path": "a.jpg"}], ItemStore)
