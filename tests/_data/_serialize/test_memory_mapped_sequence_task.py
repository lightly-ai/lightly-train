#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
from pathlib import Path

import pytest

from lightly_train._data._serialize import memory_mapped_sequence_task


class TestMemoryMappedSequenceTask:
    def test_index(self, tmp_path: Path) -> None:
        memory_mapped_sequence_task.write_filenames_to_file(
            filenames=[
                ("image1.jpg", "mask1.png"),
                ("image2.jpg", "mask2.png"),
                ("image3.jpg", "mask3.png"),
            ],
            mmap_filepath=tmp_path / "test.arrow",
            column_names=["image_filenames", "mask_filenames"],
        )
        sequence = memory_mapped_sequence_task.memory_mapped_sequence_from_file(
            mmap_filepath=tmp_path / "test.arrow",
            column_names=["image_filenames", "mask_filenames"],
        )
        assert len(sequence) == 3
        assert sequence[0] == ("image1.jpg", "mask1.png")
        assert sequence[1] == ("image2.jpg", "mask2.png")
        assert sequence[2] == ("image3.jpg", "mask3.png")
        with pytest.raises(IndexError, match="list index out of range"):
            sequence[3]

    def test_slice(self, tmp_path: Path) -> None:
        memory_mapped_sequence_task.write_filenames_to_file(
            filenames=[
                ("image1.jpg", "mask1.png"),
                ("image2.jpg", "mask2.png"),
                ("image3.jpg", "mask3.png"),
            ],
            mmap_filepath=tmp_path / "test.arrow",
            column_names=["image_filenames", "mask_filenames"],
        )
        sequence = memory_mapped_sequence_task.memory_mapped_sequence_from_file(
            mmap_filepath=tmp_path / "test.arrow",
            column_names=["image_filenames", "mask_filenames"],
        )
        assert len(sequence) == 3
        assert sequence[0:2] == [
            ("image1.jpg", "mask1.png"),
            ("image2.jpg", "mask2.png"),
        ]
        assert sequence[1:3] == [
            ("image2.jpg", "mask2.png"),
            ("image3.jpg", "mask3.png"),
        ]
        assert sequence[0:100] == [
            ("image1.jpg", "mask1.png"),
            ("image2.jpg", "mask2.png"),
            ("image3.jpg", "mask3.png"),
        ]

    def test_pickle(self, tmp_path: Path) -> None:
        memory_mapped_sequence_task.write_filenames_to_file(
            filenames=[
                ("image1.jpg", "mask1.png"),
                ("image2.jpg", "mask2.png"),
                ("image3.jpg", "mask3.png"),
            ],
            mmap_filepath=tmp_path / "test.arrow",
            column_names=["image_filenames", "mask_filenames"],
        )
        sequence = memory_mapped_sequence_task.memory_mapped_sequence_from_file(
            mmap_filepath=tmp_path / "test.arrow",
            column_names=["image_filenames", "mask_filenames"],
        )
        assert len(sequence) == 3
        copy = pickle.loads(pickle.dumps(sequence))
        assert len(copy) == 3
        assert sequence[:] == copy[:]


@pytest.mark.parametrize("chunk_size", [1, 2, 10_000])
def test_write_filenames_to_file(chunk_size: int, tmp_path: Path) -> None:
    column_names = ["image_filenames", "mask_filenames"]
    memory_mapped_sequence_task.write_filenames_to_file(
        filenames=[
            ("image1.jpg", "mask1.png"),
            ("image2.jpg", "mask2.png"),
            ("image3.jpg", "mask3.png"),
        ],
        mmap_filepath=tmp_path / "test.arrow",
        chunk_size=chunk_size,
        column_names=column_names,
    )
    sequence = memory_mapped_sequence_task.memory_mapped_sequence_from_file(
        mmap_filepath=tmp_path / "test.arrow",
        column_names=column_names,
    )
    assert len(sequence) == 3
    assert sequence[:] == [
        ("image1.jpg", "mask1.png"),
        ("image2.jpg", "mask2.png"),
        ("image3.jpg", "mask3.png"),
    ]


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
        memory_mapped_sequence_task.write_filenames_to_file(
            filenames=[
                ("image1.jpg", "mask1.png"),
                ("image2.jpg", "mask2.png"),
                ("image3.jpg", "mask3.png"),
            ],
            mmap_filepath=tmp_path / "test.arrow",
            chunk_size=chunk_size,
            column_names=["image_filenames", "mask_filenames"],
        )
