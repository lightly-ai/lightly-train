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

from lightly_train import EmbeddingFormat
from lightly_train._embedding.writers import writer_helpers
from lightly_train._embedding.writers.csv_writer import CSVWriter
from lightly_train._embedding.writers.embedding_writer import EmbeddingWriter
from lightly_train._embedding.writers.torch_writer import TorchWriter


@pytest.mark.parametrize(
    "format, expected",
    [
        (EmbeddingFormat.CSV, CSVWriter),
        (EmbeddingFormat.LIGHTLY_CSV, CSVWriter),
        (EmbeddingFormat.TORCH, TorchWriter),
    ],
)
def test_get_writer(
    format: EmbeddingFormat, expected: type[EmbeddingWriter], tmp_path: Path
) -> None:
    filepath = tmp_path / "embeddings"
    writer = writer_helpers.get_writer(format=format, filepath=filepath)
    assert isinstance(writer, expected)
