#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from lightly.models.utils import get_weight_decay_parameters
from torch.optim.optimizer import Optimizer

from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.lars_args import LARSArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.sgd_args import SGDArgs
from lightly_train._optim.trainable_modules import TrainableModules

_OPTIM_TYPE_TO_ARGS: dict[OptimizerType, type[OptimizerArgs]] = {
    AdamWArgs.type(): AdamWArgs,
    SGDArgs.type(): SGDArgs,
    LARSArgs.type(): LARSArgs,
}


def get_optimizer_type(
    optim_type: str | OptimizerType,
) -> OptimizerType:
    try:
        return OptimizerType(optim_type)
    except ValueError:
        raise ValueError(
            f"Invalid optimizer type: '{optim_type}'. Valid types are: "
            f"{[t.value for t in OptimizerType]}"
        )


def get_optimizer_args_cls(optim_type: OptimizerType) -> type[OptimizerArgs]:
    try:
        return _OPTIM_TYPE_TO_ARGS[optim_type]
    except KeyError:
        raise ValueError(
            f"Invalid optimizer type: '{optim_type}'. Valid types are: "
            f"{[t.value for t in OptimizerType]}"
        )


def get_optimizer(
    optim_args: OptimizerArgs,
    trainable_modules: TrainableModules,
    lr_scale: float,
) -> Optimizer:
    params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
        modules=trainable_modules.modules
    )
    if trainable_modules.modules_no_weight_decay is not None:
        for m in trainable_modules.modules_no_weight_decay:
            params_no_weight_decay.extend(m.parameters())

    params: list[dict[str, Any]] = [{"name": "params", "params": params_weight_decay}]
    if params_no_weight_decay:
        params.append(
            {
                "name": "params_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            }
        )
    return optim_args.get_optimizer(params=params, lr_scale=lr_scale)
