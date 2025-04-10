#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Flatten

from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train.types import Batch


class EmbeddingPredictor(LightningModule):
    """PyTorch Lightning module for "predicting" embeddings.

    This module uses the `predict_step` to extract embeddings from the given
    embedding model.

    Args:
        embedding_model: The embedding model.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        super().__init__()
        self.embedding_model = embedding_model
        self.flatten = Flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_model(x)
        x = self.flatten(x)
        return x

    def predict_step(self, batch: Batch, batch_idx: int) -> tuple[Tensor, list[str]]:
        x = batch["views"][0]
        filenames = batch["filename"]
        embeddings = self(x)
        return embeddings, filenames
