#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._embedding.writers.embedding_writer import EmbeddingWriter


class TorchWriter(EmbeddingWriter):
    """Writes embeddings to a file in torch format.

    Args:
        filepath:
            Path to file where embeddings will be saved.

    Example output:
    ```
    embeddings = torch.load("embeddings.pt")
    embeddings == {
        "filenames": ["image1.jpg", "image2.jpg", ...],
        "embeddings": torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            ...
        ])
    }
    ```

    """

    def __init__(self, filepath: Path):
        super().__init__()
        self._filepath = filepath

    @classmethod
    def is_supported_format(cls, format: EmbeddingFormat) -> bool:
        return format == EmbeddingFormat.TORCH

    def save(self, embeddings: Tensor | None, filenames: list[str]) -> None:
        torch.save(
            {
                "filenames": filenames,
                "embeddings": torch.empty(0) if embeddings is None else embeddings,
            },
            self._filepath,
        )
