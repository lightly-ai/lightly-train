#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Dict, Mapping, Protocol, Sequence, Union, overload, runtime_checkable


@runtime_checkable
class ItemStore(Protocol):
    """Storage independent view on the rows of a dataset.

    Datasets receive their rows through this protocol so that they don't have to know
    how the rows are stored. The rows are usually backed by a memory mapped Arrow
    table, but a plain list of rows works just as well.
    """

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, index: int) -> Mapping[str, str]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, str]]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Mapping[str, str], Sequence[Mapping[str, str]]]: ...

    def value_counts(self, column: str) -> Dict[str, int]:
        """Counts how often every value occurs in a column.

        Values are counted as they are stored. Callers that store multiple values in
        a single cell must expand the counts themselves.

        Returns:
            Mapping from value to number of occurrences. Empty values are not counted.
        """
        ...


class ListItemStore(Sequence[Mapping[str, str]]):
    """Item store backed by an in-memory sequence of rows."""

    def __init__(self, items: Sequence[Mapping[str, str]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> Mapping[str, str]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, str]]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Mapping[str, str], Sequence[Mapping[str, str]]]:
        return self._items[index]

    def value_counts(self, column: str) -> Dict[str, int]:
        # Imported here to keep the storage backend out of the module namespace.
        from lightly_train._data._serialize.memory_mapped_sequence import (
            as_table,
            value_counts_from_table,
        )

        counts = value_counts_from_table(table=as_table(self._items), column=column)
        return {str(value): count for value, count in counts.items()}


def as_item_store(items: Union[ItemStore, Sequence[Mapping[str, str]]]) -> ItemStore:
    """Returns `items` as an item store, wrapping it only if necessary."""
    if isinstance(items, ItemStore):
        return items
    return ListItemStore(items)
