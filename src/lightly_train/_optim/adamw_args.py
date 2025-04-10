#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import Field
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer as TorchOptimizer

from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train.types import ParamsT


class AdamWArgs(OptimizerArgs):
    lr: float = 0.001
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    eps: float = 1e-8
    weight_decay: float = 0.01

    @staticmethod
    def type() -> OptimizerType:
        return OptimizerType.ADAMW

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> TorchOptimizer:
        kwargs = self.model_dump()
        kwargs["lr"] *= lr_scale
        return AdamW(params=params, **kwargs)
