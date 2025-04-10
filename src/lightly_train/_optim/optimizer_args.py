#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.optim.optimizer import Optimizer

from lightly_train._configs.config import PydanticConfig
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train.types import ParamsT


class OptimizerArgs(PydanticConfig):
    """Base class for optimizer arguments."""

    @staticmethod
    def type() -> OptimizerType:
        """Returns the optimizer type."""
        raise NotImplementedError

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> Optimizer:
        """Returns a new optimizer instance for the given parameters.

        Args:
            params:
                Parameters to optimize.
            lr_scale:
                Learning rate scale. Will be multiplied with the learning rate in the
                optimizer.
        """
        raise NotImplementedError
