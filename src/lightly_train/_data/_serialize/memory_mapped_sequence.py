#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
from itertools import chain
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import pyarrow as pa  # type: ignore
from pyarrow import Table, ipc

logger = logging.getLogger(__name__)

Primitive = TypeVar("Primitive", bool, int, float, str)
T = Dict[str, Primitive]


def write_items_to_file(
    items: Iterable[Mapping[str, Primitive]],
    mmap_filepath: Path,
    chunk_size: int = 10_000,
) -> None:
    """Writes the filepaths to a file for memory dict."""
    if chunk_size <= 0:
        raise ValueError(f"Invalid `chunk_size` {chunk_size} must be positive!")
    logger.debug(f"Writing filepaths to '{mmap_filepath}' (chunk_size={chunk_size})")

    _stream_write_table_to_file(
        items=items,
        mmap_filepath=mmap_filepath,
        chunk_size=chunk_size,
    )


class MemoryMappedSequence(Sequence[T[Primitive]], Generic[Primitive]):
    """A memory mapped sequence built around PyArrow's memory mapped tables.

    A memory mapped sequence does not store its items in RAM but loads the data from disk.

    Pickling: A memory mapped sequence can be pickled and loaded without copying the data in
    memory. Instead, the path to the PyArrow file and the relevant column name is pickled. When
    loading a pickled memory mapped sequence, the memory map is restored from the path.

    Note: This implementation is inspired by https://github.com/huggingface/datasets. In the future
    we can add it as a hard dependency or implement table-based datasets for a richer interface.
    """

    def __init__(
        self,
        path: Path,
        columns: list[str],
    ):
        """Instantiates a new memory mapped sequence from a table and path.

        Args:
            path:
                The path to the PyArrow file.
            columns:
                The relevant columns in the table.
        """
        self._path = path
        self._columns = columns
        # The table is lazily initialized on every process independently. This avoids
        # accidentally sharing table references between processes.
        # The following scenarios are covered:
        # - Dataloader processes are created with "spawn" method:
        #   __setstate__ is called which initializes the instance again, setting the
        #   table to None.
        # - Dataloader processes are created with "fork" method:
        #   __setstate__ is not called as the dataset is not pickled. Instead the memory
        #   space for the dataset from the main process is copied. In this case, no
        #   table reference is shared as the table is re-initialized in the new process
        #   when the first item is accessed.
        self._table: Table | None = None
        # Process ID of the process that last accessed the table.
        self._pid: int | None = None

    def table(self) -> Table:
        pid = os.getpid()
        if pid != self._pid or self._table is None:
            # Re-initialize the table if the process ID has changed or if the table is
            # not initialized yet.
            self._pid = pid
            self._table = _mmap_table_from_file(mmap_filepath=self._path)
        return self._table

    def __len__(self) -> int:
        num_rows: int = self.table().num_rows
        return num_rows

    @overload
    def __getitem__(self, index: int) -> T[Primitive]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T[Primitive]]: ...

    def __getitem__(self, index: int | slice) -> T[Primitive] | Sequence[T[Primitive]]:
        if isinstance(index, int):
            rows_dict: list[T[Primitive]] = (
                self.table().select(self._columns).slice(index, 1).to_pylist()
            )
            # Each row is a dict of values corresponding to the requested columns.
            return rows_dict[0]
        else:
            start, stop, step = index.indices(len(self))

            if step == 1:
                # Contiguous slice - use slice() method (much faster)
                rows_dict = (
                    self.table()
                    .select(self._columns)
                    .slice(start, stop - start)
                    .to_pylist()
                )
            else:
                # Non-contiguous slice - still need take()
                indices = list(range(start, stop, step))
                rows_dict = self.table().select(self._columns).take(indices).to_pylist()

            return rows_dict

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self._path, "columns": self._columns}

    def __setstate__(self, state: dict[str, Any]) -> None:
        columns = state["columns"]
        path = state["path"]
        MemoryMappedSequence.__init__(self, path=path, columns=columns)

    @classmethod
    def from_file(
        cls: type[MemoryMappedSequence[Primitive]], mmap_filepath: Path
    ) -> MemoryMappedSequence[Primitive]:
        table = _mmap_table_from_file(mmap_filepath=mmap_filepath)

        num_rows = table.num_rows
        column_names = table.column_names

        logger.debug(
            f"Creating memory mapped sequence with {num_rows} '{column_names}'."
        )
        return cls(path=mmap_filepath, columns=column_names)


def _infer_type(value: Primitive) -> pa.DataType:
    # Fallback to string for any unexpected type.
    ArrowTypes = {
        bool: pa.bool_(),
        float: pa.float64(),
        int: pa.int64(),
        str: pa.string(),
    }

    return ArrowTypes.get(type(value), pa.string())


def _stream_write_table_to_file(
    items: Iterable[Mapping[str, Primitive]],
    mmap_filepath: Path,
    chunk_size: int = 10_000,
) -> None:
    it = iter(items)
    try:
        first_item = next(it)
    except StopIteration:
        # Create an empty file with no rows and no columns
        with ipc.new_file(
            sink=str(mmap_filepath.resolve()), schema=pa.schema([])
        ) as writer:
            pass
        return

    column_names = list(first_item.keys())
    schema = pa.schema([(name, _infer_type(first_item[name])) for name in column_names])

    with ipc.new_file(sink=str(mmap_filepath.resolve()), schema=schema) as writer:
        chunks: list[list[Primitive]] = list([] for _ in column_names)

        for item_count, item in enumerate(chain([first_item], it), 1):
            for chunk, name in zip(chunks, column_names):
                chunk.append(item[name])

            if item_count % chunk_size == 0:
                writer.write_table(pa.table(data=chunks, names=column_names))
                for chunk in chunks:
                    chunk.clear()

        if any(chunks):  # Check if any chunk has remaining data
            writer.write_table(pa.table(data=chunks, names=column_names))


def _mmap_table_from_file(mmap_filepath: Path) -> Table:
    with pa.memory_map(str(mmap_filepath.resolve())) as source:
        return ipc.open_file(source).read_all()
