#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.nn import Module

from lightly_train._configs.config import PydanticConfig
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._scaling import ScalingInfo


class MethodArgs(PydanticConfig):
    """Arguments for a method.

    This does not include optimizer or scheduler arguments.
    """

    pass

    def resolve_auto(
        self, scaling_info: ScalingInfo, optimizer_args: OptimizerArgs, model: Module
    ) -> None:
        """Resolves all fields with the value 'auto' to their actual value."""
        pass
