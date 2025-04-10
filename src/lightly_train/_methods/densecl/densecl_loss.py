#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.nn import functional as F


class DenseCLLoss(Module):
    """DenseCLLoss implementation.

    Uses 'out0' and 'out1' for positives and 'negatives' for negatives.

    TODO(Philipp, 11/24): Move this to LightlySSL once we're certain DenseCL works.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = CrossEntropyLoss(reduction="mean")

    def forward(self, out0: Tensor, out1: Tensor, negatives: Tensor) -> Tensor:
        # Normalize the output to length 1
        out0 = F.normalize(out0, dim=-1)
        out1 = F.normalize(out1, dim=-1)

        # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
        # of the i-th sample in the batch to its positive pair
        sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

        # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
        # of the i-th sample to the j-th negative sample
        sim_neg = torch.einsum("nc,ck->nk", out0, negatives)

        # Set the labels to maximize sim_pos in relation to sim_neg
        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], device=out0.device, dtype=torch.long)

        # Calculate the cross-entropy loss
        loss: Tensor = self.cross_entropy(logits, labels)
        return loss
