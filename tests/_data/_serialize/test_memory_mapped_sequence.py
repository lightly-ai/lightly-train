#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
from pathlib import Path

import pyarrow as pa  # type: ignore
import pytest
from pyarrow import ipc

from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import MemoryMappedSequence


class TestMemoryMappedSequence:
    def test_index(self, tmp_path: Path) -> None:
        memory_mapped_sequence.write_filenames_to_file(
            filenames=["image1.jpg", "image2.jpg", "image3.jpg"],
            mmap_filepath=tmp_path / "test.arrow",
        )
        sequence = memory_mapped_sequence.memory_mapped_sequence_from_file(
            mmap_filepath=tmp_path / "test.arrow",
        )
        assert len(sequence) == 3
        assert sequence[0] == "image1.jpg"
        assert sequence[1] == "image2.jpg"
        assert sequence[2] == "image3.jpg"
        with pytest.raises(IndexError, match="index out of bounds"):
            sequence[3]

    def test_slice(self, tmp_path: Path) -> None:
        memory_mapped_sequence.write_filenames_to_file(
            filenames=["image1.jpg", "image2.jpg", "image3.jpg"],
            mmap_filepath=tmp_path / "test.arrow",
        )
        sequence = memory_mapped_sequence.memory_mapped_sequence_from_file(
            mmap_filepath=tmp_path / "test.arrow",
        )
        assert len(sequence) == 3
        assert sequence[0:2] == ["image1.jpg", "image2.jpg"]
        assert sequence[1:3] == ["image2.jpg", "image3.jpg"]
        assert sequence[0:100] == ["image1.jpg", "image2.jpg", "image3.jpg"]

    def test_pickle(self, tmp_path: Path) -> None:
        memory_mapped_sequence.write_filenames_to_file(
            filenames=["image1.jpg", "image2.jpg", "image3.jpg"],
            mmap_filepath=tmp_path / "test.arrow",
        )
        sequence = memory_mapped_sequence.memory_mapped_sequence_from_file(
            mmap_filepath=tmp_path / "test.arrow",
        )
        assert len(sequence) == 3
        copy = pickle.loads(pickle.dumps(sequence))
        assert len(copy) == 3
        assert sequence[:] == copy[:]

    def test_multicolumn(self, tmp_path: Path) -> None:
        # Create a custom table with multiple columns of different types.
        mmap_filepath = tmp_path / "test.arrow"
        schema = pa.schema(
            [
                ("column1", pa.string()),
                ("column2", pa.int64()),
                ("column3", pa.float64()),
            ]
        )
        with ipc.new_file(sink=str(mmap_filepath.resolve()), schema=schema) as writer:
            writer.write_table(
                pa.table(
                    {
                        "column1": pa.array(["hello", "world"]),
                        "column2": pa.array([1, 2]),
                        "column3": pa.array([0.1, 0.2]),
                    }
                )
            )
        # Create a sequence from each column.
        str_sequence: MemoryMappedSequence[str] = MemoryMappedSequence(
            path=mmap_filepath, column="column1"
        )
        assert str_sequence[:] == ["hello", "world"]
        int_sequence: MemoryMappedSequence[int] = MemoryMappedSequence(
            path=mmap_filepath, column="column2"
        )
        assert int_sequence[:] == [1, 2]
        float_sequence: MemoryMappedSequence[float] = MemoryMappedSequence(
            path=mmap_filepath, column="column3"
        )
        assert float_sequence[:] == [0.1, 0.2]


@pytest.mark.parametrize("chunk_size", [1, 2, 10_000])
@pytest.mark.parametrize("column_name", ["", "hi"])
def test_write_filenames_to_file(
    chunk_size: int, column_name: str, tmp_path: Path
) -> None:
    memory_mapped_sequence.write_filenames_to_file(
        filenames=["image1.jpg", "image2.jpg", "image3.jpg"],
        mmap_filepath=tmp_path / "test.arrow",
        chunk_size=chunk_size,
        column_name=column_name,
    )
    sequence = memory_mapped_sequence.memory_mapped_sequence_from_file(
        mmap_filepath=tmp_path / "test.arrow",
        column_name=column_name,
    )
    assert len(sequence) == 3
    assert sequence[:] == ["image1.jpg", "image2.jpg", "image3.jpg"]


@pytest.mark.parametrize(
    "chunk_size",
    [0, -1],
)
def test_write_filenames_to_file__invalid_chunks(
    chunk_size: int, tmp_path: Path
) -> None:
    with pytest.raises(
        ValueError, match=f"Invalid `chunk_size` {chunk_size} must be positive!"
    ):
        memory_mapped_sequence.write_filenames_to_file(
            filenames=["image1.jpg", "image2.jpg", "image3.jpg"],
            mmap_filepath=tmp_path / "test.arrow",
            chunk_size=chunk_size,
        )
