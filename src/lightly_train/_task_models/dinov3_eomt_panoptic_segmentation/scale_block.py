#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Modifications Copyright 2025 Lightly AG:
# - Replace timm LayerNorm2D implementation with torch version

from __future__ import annotations

from typing import Type

from torch import Tensor, nn
from torch.nn import LayerNorm, Module


class ScaleBlock(Module):
    def __init__(self, embed_dim: int, conv1_layer: Type[Module] = nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2D(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x


class LayerNorm2D(LayerNorm):
    def __init__(self, embed_dim: int):
        super().__init__(normalized_shape=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x
