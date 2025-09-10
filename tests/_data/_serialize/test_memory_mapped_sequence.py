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
