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
from lightly_train._embedding.writers.torch_writer import TorchWriter


class TestTorchWriter:
    @pytest.mark.parametrize(
        "format, expected",
        [
            (EmbeddingFormat.TORCH, True),
            (EmbeddingFormat.CSV, False),
        ],
    )
    def test_is_supported_format(self, format: EmbeddingFormat, expected: bool) -> None:
        assert TorchWriter.is_supported_format(format=format) is expected

    def test_save(self, tmp_path: Path) -> None:
        filepath = tmp_path / "embeddings.pt"
        writer = TorchWriter(filepath=filepath)
        embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        filenames = ["file1.png", "file , special.png"]
        writer.save(embeddings=embeddings, filenames=filenames)

        loaded = torch.load(filepath)
        assert loaded["filenames"] == filenames
        assert torch.equal(loaded["embeddings"], embeddings)

    def test_save__none(self, tmp_path: Path) -> None:
        filepath = tmp_path / "embeddings.pt"
        writer = TorchWriter(filepath=filepath)
        writer.save(embeddings=None, filenames=[])
        loaded = torch.load(filepath)
        assert loaded["filenames"] == []
        assert loaded["embeddings"].shape == (0,)
