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

from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import (
    MemoryMappedSequence,
)


class TestMemoryMappedSequence:
    def test_index(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        memory_mapped_sequence.write_items_to_file(
            items=[
                {
                    "image_filepaths": str(image_dir / "image1.jpg"),
                    "mask_filepaths": str(mask_dir / "mask1.png"),
                },
                {
                    "image_filepaths": str(image_dir / "image2.jpg"),
                    "mask_filepaths": str(mask_dir / "mask2.png"),
                },
                {
                    "image_filepaths": str(image_dir / "image3.jpg"),
                    "mask_filepaths": str(mask_dir / "mask3.png"),
                },
            ],
            mmap_filepath=tmp_path / "test.arrow",
        )
        sequence = MemoryMappedSequence[str].from_file(
            mmap_filepath=tmp_path / "test.arrow",
        )
        assert len(sequence) == 3
        assert sequence[0] == {
            "image_filepaths": str(image_dir / "image1.jpg"),
            "mask_filepaths": str(mask_dir / "mask1.png"),
        }
        assert sequence[1] == {
            "image_filepaths": str(image_dir / "image2.jpg"),
            "mask_filepaths": str(mask_dir / "mask2.png"),
        }
        assert sequence[2] == {
            "image_filepaths": str(image_dir / "image3.jpg"),
            "mask_filepaths": str(mask_dir / "mask3.png"),
        }
        with pytest.raises(IndexError, match="list index out of range"):
            sequence[3]

    def test_slice(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        memory_mapped_sequence.write_items_to_file(
            items=[
                {
                    "image_filepaths": str(image_dir / "image1.jpg"),
                    "mask_filepaths": str(mask_dir / "mask1.png"),
                },
                {
                    "image_filepaths": str(image_dir / "image2.jpg"),
                    "mask_filepaths": str(mask_dir / "mask2.png"),
                },
                {
                    "image_filepaths": str(image_dir / "image3.jpg"),
                    "mask_filepaths": str(mask_dir / "mask3.png"),
                },
            ],
            mmap_filepath=tmp_path / "test.arrow",
        )
        sequence = MemoryMappedSequence[str].from_file(
            mmap_filepath=tmp_path / "test.arrow",
        )
        assert len(sequence) == 3
        assert sequence[0:2] == [
            {
                "image_filepaths": str(image_dir / "image1.jpg"),
                "mask_filepaths": str(mask_dir / "mask1.png"),
            },
            {
                "image_filepaths": str(image_dir / "image2.jpg"),
                "mask_filepaths": str(mask_dir / "mask2.png"),
            },
        ]
        assert sequence[1:3] == [
            {
                "image_filepaths": str(image_dir / "image2.jpg"),
                "mask_filepaths": str(mask_dir / "mask2.png"),
            },
            {
                "image_filepaths": str(image_dir / "image3.jpg"),
                "mask_filepaths": str(mask_dir / "mask3.png"),
            },
        ]
        assert sequence[0:100] == [
            {
                "image_filepaths": str(image_dir / "image1.jpg"),
                "mask_filepaths": str(mask_dir / "mask1.png"),
            },
            {
                "image_filepaths": str(image_dir / "image2.jpg"),
                "mask_filepaths": str(mask_dir / "mask2.png"),
            },
            {
                "image_filepaths": str(image_dir / "image3.jpg"),
                "mask_filepaths": str(mask_dir / "mask3.png"),
            },
        ]

    def test_pickle(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        memory_mapped_sequence.write_items_to_file(
            items=[
                {
                    "image_filepaths": str(image_dir / "image1.jpg"),
                    "mask_filepaths": str(mask_dir / "mask1.png"),
                },
                {
                    "image_filepaths": str(image_dir / "image2.jpg"),
                    "mask_filepaths": str(mask_dir / "mask2.png"),
                },
                {
                    "image_filepaths": str(image_dir / "image3.jpg"),
                    "mask_filepaths": str(mask_dir / "mask3.png"),
                },
            ],
            mmap_filepath=tmp_path / "test.arrow",
        )
        sequence = MemoryMappedSequence[str].from_file(
            mmap_filepath=tmp_path / "test.arrow",
        )
        assert len(sequence) == 3
        copy = pickle.loads(pickle.dumps(sequence))
        assert len(copy) == 3
        assert sequence[:] == copy[:]


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 10_000])
def test_write_items_to_file(chunk_size: int, tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    memory_mapped_sequence.write_items_to_file(
        items=[
            {
                "image_filepaths": str(image_dir / "image1.jpg"),
                "mask_filepaths": str(mask_dir / "mask1.png"),
            },
            {
                "image_filepaths": str(image_dir / "image2.jpg"),
                "mask_filepaths": str(mask_dir / "mask2.png"),
            },
            {
                "image_filepaths": str(image_dir / "image3.jpg"),
                "mask_filepaths": str(mask_dir / "mask3.png"),
            },
        ],
        mmap_filepath=tmp_path / "test.arrow",
        chunk_size=chunk_size,
    )
    sequence = MemoryMappedSequence[str].from_file(
        mmap_filepath=tmp_path / "test.arrow",
    )
    assert len(sequence) == 3
    assert sequence[:] == [
        {
            "image_filepaths": str(image_dir / "image1.jpg"),
            "mask_filepaths": str(mask_dir / "mask1.png"),
        },
        {
            "image_filepaths": str(image_dir / "image2.jpg"),
            "mask_filepaths": str(mask_dir / "mask2.png"),
        },
        {
            "image_filepaths": str(image_dir / "image3.jpg"),
            "mask_filepaths": str(mask_dir / "mask3.png"),
        },
    ]


def _write_and_read(
    items: list[dict[str, str]], tmp_path: Path, chunk_size: int = 10_000
) -> MemoryMappedSequence[str]:
    memory_mapped_sequence.write_items_to_file(
        items=items, mmap_filepath=tmp_path / "test.arrow", chunk_size=chunk_size
    )
    return MemoryMappedSequence[str].from_file(mmap_filepath=tmp_path / "test.arrow")


class TestValueCounts:
    @pytest.mark.parametrize("chunk_size", [1, 2, 10_000])
    def test_delimiter(self, chunk_size: int, tmp_path: Path) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": "3,12"},
            {"image_path": "b.jpg", "class_id": "3"},
            {"image_path": "c.jpg", "class_id": "12"},
            {"image_path": "d.jpg", "class_id": "12"},
        ]
        sequence = _write_and_read(items, tmp_path, chunk_size=chunk_size)
        assert memory_mapped_sequence.value_counts(
            sequence, column="class_id", delimiter=","
        ) == {"3": 2, "12": 3}

    def test_no_delimiter(self, tmp_path: Path) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": "3,12"},
            {"image_path": "b.jpg", "class_id": "3,12"},
            {"image_path": "c.jpg", "class_id": "3"},
        ]
        sequence = _write_and_read(items, tmp_path)
        # Without a delimiter the full value is counted.
        assert memory_mapped_sequence.value_counts(sequence, column="class_id") == {
            "3,12": 2,
            "3": 1,
        }

    def test_whitespace_and_empty_values(self, tmp_path: Path) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": " 3 , 12"},
            {"image_path": "b.jpg", "class_id": ""},
            {"image_path": "c.jpg", "class_id": "3,"},
        ]
        sequence = _write_and_read(items, tmp_path)
        # Whitespace is trimmed and empty values are not counted.
        assert memory_mapped_sequence.value_counts(
            sequence, column="class_id", delimiter=","
        ) == {"3": 2, "12": 1}

    def test_repeated_value_in_one_row(self, tmp_path: Path) -> None:
        items = [{"image_path": "a.jpg", "class_id": "3,3"}]
        sequence = _write_and_read(items, tmp_path)
        # Occurrences are counted, they are not deduplicated per row.
        assert memory_mapped_sequence.value_counts(
            sequence, column="class_id", delimiter=","
        ) == {"3": 2}

    def test_no_items(self, tmp_path: Path) -> None:
        # An empty sequence is written without any columns.
        sequence = _write_and_read([], tmp_path)
        assert (
            memory_mapped_sequence.value_counts(
                sequence, column="class_id", delimiter=","
            )
            == {}
        )

    def test_unknown_column(self, tmp_path: Path) -> None:
        sequence = _write_and_read([{"image_path": "a.jpg"}], tmp_path)
        assert memory_mapped_sequence.value_counts(sequence, column="class_id") == {}

    def test_list_items(self, tmp_path: Path) -> None:
        items = [
            {"image_path": "a.jpg", "class_id": "3,12"},
            {"image_path": "b.jpg", "class_id": "3"},
        ]
        sequence = _write_and_read(items, tmp_path)
        # Sequences that are not memory mapped give the same result.
        assert memory_mapped_sequence.value_counts(
            items, column="class_id", delimiter=","
        ) == memory_mapped_sequence.value_counts(
            sequence, column="class_id", delimiter=","
        )
