#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import Module

from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt


class DINOv3ConvNextWrapper(Module):
    def __init__(self, model: ConvNeXt) -> None:
        super().__init__()
        self.backbone = model

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        feats = self.backbone.get_intermediate_layers(x, n=3, reshape=True)
        assert isinstance(feats, tuple)
        assert all(isinstance(f, Tensor) for f in feats)
        return feats
