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

from torch import Tensor
from torch.nn import Module

from lightly_train import _torch_helpers
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper

logger = logging.getLogger(__name__)


class DINOv3ConvNextWrapper(Module):
    def __init__(self, model_wrapper: DINOv3VConvNeXtModelWrapper) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = model_wrapper.get_model().patch_size
        _torch_helpers.register_load_state_dict_pre_hook(
            self, DINOv3ConvNextWrapper._remap_legacy_keys
        )

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper.get_model()

    @staticmethod
    def _remap_legacy_keys(
        module: Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        old_subprefix = prefix + "backbone."
        new_subprefix = prefix + "_model_wrapper._model."
        if any(k.startswith(old_subprefix) for k in state_dict):
            logger.info(
                "Detected old DINOv3ConvNextWrapper checkpoint format "
                "(backbone. → _model_wrapper._model.). Remapping keys."
            )
            for k in [
                k for k in list(state_dict.keys()) if k.startswith(old_subprefix)
            ]:
                state_dict[new_subprefix + k[len(old_subprefix) :]] = state_dict.pop(k)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        feats = self._model_wrapper.forward_multiscale_features(x, [1, 2, 3])
        return tuple(f["features"] for f in feats)
