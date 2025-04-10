#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.optim.sgd import SGD

from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train.types import ParamsT


class SGDArgs(OptimizerArgs):
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001

    @staticmethod
    def type() -> OptimizerType:
        return OptimizerType.SGD

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> TorchOptimizer:
        kwargs = self.model_dump()
        kwargs["lr"] *= lr_scale
        return SGD(params=params, **kwargs)
