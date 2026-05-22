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

from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper


class DINOv3ConvNextWrapper(Module):
    def __init__(self, model_wrapper: DINOv3VConvNeXtModelWrapper) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = model_wrapper._model.patch_size

    @property
    def backbone_model(self):
        return self._model_wrapper._model

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        feats = self._model_wrapper.forward_multiscale_features(x, [1, 2, 3])
        return tuple(f["features"] for f in feats)
