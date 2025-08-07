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
from pathlib import Path
from typing import Any, Generic, Iterable, Sequence, TypeVar, overload

import pyarrow as pa  # type: ignore
from pyarrow import Table, ipc

logger = logging.getLogger(__name__)

T = TypeVar("T")


def write_filenames_to_file(
    filenames: Iterable[str],
    mmap_filepath: Path,
    chunk_size: int = 10_000,
    column_name: str = "filenames",
) -> None:
    """Writes the filenames to a file for memory mapping."""
    if chunk_size <= 0:
        raise ValueError(f"Invalid `chunk_size` {chunk_size} must be positive!")
    logger.debug(f"Writing filenames to '{mmap_filepath}' (chunk_size={chunk_size})")
    _stream_write_table_to_file(
        items=filenames,
        mmap_filepath=mmap_filepath,
        chunk_size=chunk_size,
        column_name=column_name,
    )


def memory_mapped_sequence_from_file(
    mmap_filepath: Path, column_name: str = "filenames"
) -> MemoryMappedSequence[str]:
    table = _mmap_table_from_file(mmap_filepath=mmap_filepath)
    logger.debug(
        f"Creating memory mapped sequence with {table.num_rows} '{column_name}'."
    )
    return MemoryMappedSequence(path=mmap_filepath, column=column_name)


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
        path: Path,
        column: str,
    ):
        """Instantiates a new memory mapped sequence from a table and path.

        Args:
            path:
                The path to the PyArrow file.
            column:
                The relevant column in the table.
        """
        self._path = path
        self._column = column
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
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, int):
            item_: T = self.table().column(self._column)[index].as_py()
            return item_
        else:
            items: Sequence[T] = self.table().column(self._column)[index].to_pylist()
            return items

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self._path, "column": self._column}

    def __setstate__(self, state: dict[str, Any]) -> None:
        column = state["column"]
        path = state["path"]
        MemoryMappedSequence.__init__(self, path=path, column=column)


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
