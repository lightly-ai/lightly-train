#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generic, Iterable, Sequence, TypeVar, overload

import pyarrow as pa  # type: ignore
from pyarrow import Table, ipc

logger = logging.getLogger(__name__)

T = TypeVar("T")


def memory_mapped_sequence_from_filenames(
    filenames: Iterable[str],
    mmap_filepath: Path,
    chunk_size: int = 10_000,
    column_name: str = "filenames",
) -> MemoryMappedSequence[str]:
    """Creates and returns a memory mapped sequence from the iterable filenames.

    Args:
        filenames:
            Iterable of filenames.
        mmap_filepath:
            Path to the PyArrow file used for the memory map.
        chunk_size:
            Number of filenames written to the file at once.
        column:
            The column name in which filenames are stored.

    Returns:
        A memory mapped sequence of filenames.
    """

    if chunk_size <= 0:
        raise ValueError(f"Invalid `chunk_size` {chunk_size} must be positive!")
    logger.debug(f"Writing filenames to '{mmap_filepath}' (chunk_size={chunk_size})")
    _stream_write_table_to_file(
        items=filenames,
        mmap_filepath=mmap_filepath,
        chunk_size=chunk_size,
        column_name=column_name,
    )
    return memory_mapped_sequence_from_file(
        mmap_filepath=mmap_filepath, column_name=column_name
    )


def memory_mapped_sequence_from_file(
    mmap_filepath: Path, column_name: str = "filenames"
) -> MemoryMappedSequence[str]:
    table = _mmap_table_from_file(mmap_filepath=mmap_filepath)
    logger.debug(
        f"Creating memory mapped sequence with {table.num_rows} '{column_name}'."
    )
    return MemoryMappedSequence(table=table, path=mmap_filepath, column=column_name)


class MemoryMappedSequence(Sequence[T], Generic[T]):
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
        table: Table,
        path: Path,
        column: str,
    ):
        """Instantiates a new memory mapped sequence from a table and path.

        Args:
            table:
                The PyArrow table.
            path:
                The path to the PyArrow file.
            column:
                The relevant column in the table.
        """
        self._table = table
        self._path = path
        self._column = column

    def __len__(self) -> int:
        num_rows: int = self._table.num_rows
        return num_rows

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, int):
            item_: T = self._table.column(self._column)[index].as_py()
            return item_
        else:
            items: Sequence[T] = self._table.column(self._column)[index].to_pylist()
            return items

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self._path, "column": self._column}

    def __setstate__(self, state: dict[str, Any]) -> None:
        column = state["column"]
        path = state["path"]
        table = _mmap_table_from_file(path)
        MemoryMappedSequence.__init__(self, table=table, path=path, column=column)


def _stream_write_table_to_file(
    items: Iterable[T],
    mmap_filepath: Path,
    chunk_size: int = 10_000,
    column_name: str = "items",
) -> None:
    schema = pa.schema([(column_name, pa.string())])
    with ipc.new_file(sink=str(mmap_filepath.resolve()), schema=schema) as writer:
        chunk = []
        for item in items:
            chunk.append(item)
            if len(chunk) == chunk_size:
                writer.write_table(pa.table({column_name: pa.array(chunk)}))
                chunk.clear()
        if len(chunk) > 0:
            writer.write_table(pa.table({column_name: pa.array(chunk)}))


def _mmap_table_from_file(mmap_filepath: Path) -> Table:
    with pa.memory_map(str(mmap_filepath.resolve())) as source:
        return ipc.open_file(source).read_all()
