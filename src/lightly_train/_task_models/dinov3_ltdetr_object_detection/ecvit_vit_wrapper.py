#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

from torch import Tensor
from torch.nn import Module

from lightly_train._models.ecvit.ecvit import ECViTWrapper

logger = logging.getLogger(__name__)


class ECViTBackboneWrapper(Module):
    """Thin adapter that exposes an :class:`ECViTWrapper` to the LTDETR task model.

    The ECViT wrapper already returns ``(P3, P4, P5)`` (strides 8/16/32) with the
    correct per-level channel counts (matching the DINOv3 ViT LTDETR encoder
    configs by ``proj_dim``), so this class is a pass-through that:

    - stores ``patch_size`` (fixed at 16) for the train/val transforms,
    - exposes ``backbone_model`` returning the wrapped ECViT wrapper,
    - delegates ``forward`` to the wrapped wrapper.

    Unlike :class:`DINOv3STAs`, this wrapper does not register a state-dict
    pre-hook: there are no upstream ECViT-LTDETR checkpoints to migrate, and
    :class:`ECViTWrapper` already loads its own backbone weights via
    :meth:`ECViTWrapper._load_backbone_weights` during construction.
    """

    # ECViT uses a ConvPyramidPatchEmbed with a fixed patch size of 16 (see
    # ``ECViTWrapper.__init__``); the wrapper itself does not expose a true
    # ``patch_size`` attribute, so we hard-code 16 here.
    _ECVIT_PATCH_SIZE = 16

    def __init__(self, model_wrapper: ECViTWrapper) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper
        self.patch_size = self._ECVIT_PATCH_SIZE

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self._model_wrapper(x)  # type: ignore[no-any-return]
