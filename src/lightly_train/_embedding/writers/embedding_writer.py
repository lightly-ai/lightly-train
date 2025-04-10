#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Sequence

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.prediction_writer import WriteInterval
from torch import Tensor

from lightly_train._embedding.embedding_format import EmbeddingFormat


class EmbeddingWriter(BasePredictionWriter):
    """PyTorch Lightning callback for writing embeddings."""

    def __init__(self) -> None:
        # Ignore "str" has no attribute "value" type error for string enum.
        super().__init__(write_interval=WriteInterval.BATCH.value)  # type: ignore[attr-defined]
        self._embeddings: list[Tensor] = []
        self._filenames: list[str] = []

    @property
    def embeddings(self) -> Tensor | None:
        if self._embeddings:
            return torch.cat(self._embeddings)
        return None

    @property
    def filenames(self) -> list[str]:
        return self._filenames

    @classmethod
    @abstractmethod
    def is_supported_format(cls, format: EmbeddingFormat) -> bool:
        """Returns True if the writer supports the given format."""
        raise NotImplementedError

    @abstractmethod
    def save(self, embeddings: Tensor | None, filenames: list[str]) -> None:
        """Save embeddings."""
        raise NotImplementedError

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        embeddings, filenames = prediction
        self._embeddings.append(embeddings.detach().cpu())
        self._filenames.extend(filenames)

    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._reset()

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.save(embeddings=self.embeddings, filenames=self.filenames)
        self._reset()

    def _reset(self) -> None:
        self._embeddings = []
        self._filenames = []
