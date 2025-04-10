#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._embedding.writers.csv_writer import CSVWriter


class TestCSVWriter:
    @pytest.mark.parametrize(
        "format, expected",
        [
            (EmbeddingFormat.CSV, True),
            (EmbeddingFormat.LIGHTLY_CSV, True),
            (EmbeddingFormat.TORCH, False),
        ],
    )
    def test_is_supported_format(self, format: EmbeddingFormat, expected: bool) -> None:
        assert CSVWriter.is_supported_format(format=format) is expected

    def test_save__csv(self, tmp_path: Path) -> None:
        filepath = tmp_path / "embeddings.csv"
        writer = CSVWriter(filepath=filepath, format=EmbeddingFormat.CSV)
        embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        writer.save(
            embeddings=embeddings, filenames=["file1.png", "file , special.png"]
        )
        assert filepath.read_text() == (
            "filename,embedding_0,embedding_1\n"
            "file1.png,1.0,2.0\n"
            '"file , special.png",3.0,4.0\n'
        )

    def test_save__csv__none(self, tmp_path: Path) -> None:
        filepath = tmp_path / "embeddings.csv"
        writer = CSVWriter(filepath=filepath, format=EmbeddingFormat.CSV)
        writer.save(embeddings=None, filenames=[])
        assert filepath.read_text() == ("filename\n")

    def test_save__lightly_csv(self, tmp_path: Path) -> None:
        filepath = tmp_path / "embeddings.csv"
        writer = CSVWriter(filepath=filepath, format=EmbeddingFormat.LIGHTLY_CSV)
        embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        writer.save(
            embeddings=embeddings, filenames=["file1.png", "file , special.png"]
        )
        assert filepath.read_text() == (
            "filenames,embedding_0,embedding_1,labels\n"
            "file1.png,1.0,2.0,0\n"
            '"file , special.png",3.0,4.0,0\n'
        )

    def test_save__lightly_csv__none(self, tmp_path: Path) -> None:
        filepath = tmp_path / "embeddings.csv"
        writer = CSVWriter(filepath=filepath, format=EmbeddingFormat.LIGHTLY_CSV)
        writer.save(embeddings=None, filenames=[])
        assert filepath.read_text() == ("filenames,labels\n")
