#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import cast

from lightly.utils.lars import LARS
from torch.optim.optimizer import Optimizer as TorchOptimizer

from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train.types import ParamsT


class LARSArgs(OptimizerArgs):
    lr: float = 0.3
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False
    trust_coefficient: float = 0.001
    eps: float = 1e-8

    @staticmethod
    def type() -> OptimizerType:
        return OptimizerType.LARS

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> TorchOptimizer:
        kwargs = self.model_dump()
        kwargs["lr"] *= lr_scale
        return cast(TorchOptimizer, LARS(params=params, **kwargs))
