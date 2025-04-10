#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from lightly.utils import io as lightly_io
from torch import Tensor

from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._embedding.writers.embedding_writer import EmbeddingWriter


class CSVWriter(EmbeddingWriter):
    """Writes embeddings to a CSV file.

    Args:
        filepath:
            Path to the CSV file.
        format:
            Format of the CSV file. Valid formats are 'csv' and 'lightly_csv'.
            Use 'lightly_csv' if you want to use the embeddings as custom embeddings
            with the Lightly Worker. See the relevant docs for more information:
            https://docs.lightly.ai/docs/custom-embeddings

    Example output for 'csv' format:
    ```
    filename,embedding_0,embedding_1,embedding_2
    image1.jpg,0.1,0.2,0.3
    image2.jpg,0.4,0.5,0.6
    ...
    ```

    Example output for 'lightly_csv' format:
    ```
    filenames,embedding_0,embedding_1,embedding_2,labels
    image1.jpg,0.1,0.2,0.3,0
    image2.jpg,0.4,0.5,0.6,0
    ...
    ```
    """

    _SUPPORTED_FORMATS = (EmbeddingFormat.CSV, EmbeddingFormat.LIGHTLY_CSV)

    def __init__(self, filepath: Path, format: EmbeddingFormat):
        super().__init__()
        if not self.is_supported_format(format):
            raise ValueError(
                f"Unsupported embedding format: '{format}', supported formats are "
                f"{self._SUPPORTED_FORMATS}"
            )
        self._format = format
        self._filepath = filepath

    @classmethod
    def is_supported_format(cls, format: EmbeddingFormat) -> bool:
        return format in cls._SUPPORTED_FORMATS

    def save(self, embeddings: Tensor | None, filenames: list[str]) -> None:
        """Saves the embeddings to file."""
        if self._format == EmbeddingFormat.CSV:
            save_csv(
                filepath=self._filepath, embeddings=embeddings, filenames=filenames
            )
        elif self._format == EmbeddingFormat.LIGHTLY_CSV:
            save_lightly_csv(
                filepath=self._filepath, embeddings=embeddings, filenames=filenames
            )


def save_csv(filepath: Path, embeddings: Tensor | None, filenames: list[str]) -> None:
    embs = [] if embeddings is None else embeddings.tolist()
    if len(embs) != len(filenames):
        raise ValueError(
            f"Number of embeddings ({len(embs)}) does not match number of filenames "
            "({len(filenames)})"
        )
    header = ["filename"]
    if embs:
        header.extend([f"embedding_{i}" for i in range(len(embs[0]))])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(header)
        for filename, emb in zip(filenames, embs):
            writer.writerow([filename] + emb)


def save_lightly_csv(
    filepath: Path, embeddings: Tensor | None, filenames: list[str]
) -> None:
    embs = (
        np.empty((0, 0), np.float64)
        if embeddings is None
        else np.array(embeddings, dtype=np.float64)
    )
    if len(embs) != len(filenames):
        raise ValueError(
            f"Number of embeddings ({len(embs)}) does not match number of filenames "
            f"({len(filenames)})"
        )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lightly_io.save_embeddings(
        path=str(filepath),
        embeddings=embs,
        labels=[0] * len(filenames),
        filenames=filenames,
    )
