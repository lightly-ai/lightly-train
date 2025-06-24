#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# References:
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
#
# Modifications Copyright (c) Lightly AG and affiliates:
#   - Add type hints to the functions
#   - Modify imports to follow Lightly's conventions


from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import (
    GELU,
    BatchNorm1d,
    Linear,
    Module,
    Sequential,
    functional,
    init,
)
from torch.nn.utils import parametrizations


class DINOv2ProjectionHead(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
    ) -> None:
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers=nlayers,
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        self.last_layer = parametrizations.weight_norm(
            Linear(bottleneck_dim, out_dim, bias=False)
        )
        # original0 is weight_g, see: https://github.com/pytorch/pytorch/blob/7bcf7da3a268b435777fe87c7794c382f444e86d/torch/nn/utils/parametrizations.py#L355-L361
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, m: Module) -> None:
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int,
    use_bn: bool = False,
    bias: bool = True,
) -> Module:
    if nlayers == 1:
        return Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers: list[Module] = [Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(BatchNorm1d(hidden_dim))
        layers.append(GELU())
        for _ in range(nlayers - 2):
            layers.append(Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(BatchNorm1d(hidden_dim))
            layers.append(GELU())
        layers.append(Linear(hidden_dim, bottleneck_dim, bias=bias))
        return Sequential(*layers)
