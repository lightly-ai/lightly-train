#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper

logger = logging.getLogger(__name__)


class DINOv3ConvNextWrapper(Module):
    def __init__(self, model_wrapper: DINOv3VConvNeXtModelWrapper) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = model_wrapper._model.patch_size

    @property
    def backbone_model(self):
        return self._model_wrapper._model

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            old_prefix = "backbone."
            new_prefix = "_model_wrapper._model."
            if any(k.startswith(old_prefix) for k in state_dict):
                logger.info(
                    "Detected old DINOv3ConvNextWrapper checkpoint format "
                    "(backbone. → _model_wrapper._model.). Remapping keys."
                )
                remapped = {}
                for k, v in state_dict.items():
                    if k.startswith(old_prefix):
                        k = new_prefix + k[len(old_prefix) :]
                    remapped[k] = v
                try:
                    return super().load_state_dict(
                        remapped, strict=strict, assign=assign
                    )
                except RuntimeError:
                    pass
            raise

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        feats = self._model_wrapper.forward_multiscale_features(x, [1, 2, 3])
        return tuple(f["features"] for f in feats)
