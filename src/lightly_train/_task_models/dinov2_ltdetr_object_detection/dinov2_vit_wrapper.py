#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Sequence, Tuple

from torch import Tensor
from torch.nn import Module

from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)


class DINOv2ViTWrapper(Module):
    # TODO: Lionel(09/25) Try the DEIMv2 wrapper: https://github.com/Intellindust-AI-Lab/DEIMv2/blob/main/engine/backbone/dinov3_adapter.py#L72
    def __init__(
        self, model: DinoVisionTransformer, keep_indices: Sequence[int] = (5, 8, 11)
    ):
        super().__init__()
        self.keep_indices = list(keep_indices)
        self.backbone = model

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        # TODO: Lionel(09/25) Infer minimum n from keep_indices.
        feats = self.backbone.get_intermediate_layers(x, n=12, reshape=True)
        feats_: list[Tensor] = [feats[i] for i in self.keep_indices]  # type: ignore[misc]
        assert all(isinstance(f, Tensor) for f in feats_)
        return tuple(feats_)
