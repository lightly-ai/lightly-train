#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch.nn import Embedding, EmbeddingBag, Module, Parameter
from torch.nn.modules import CrossMapLRN2d, GroupNorm, LayerNorm, LocalResponseNorm
from torch.nn.modules.batchnorm import _NormBase
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


_NORM_LAYERS = (_NormBase, LayerNorm, CrossMapLRN2d, LocalResponseNorm, GroupNorm)


def get_weight_decay_parameters(
    modules: Iterable[Module],
    extra_no_decay_keys: Iterable[str] | None = None,
    norm_layers: tuple[type[Module], ...] = _NORM_LAYERS,
) -> tuple[list[Parameter], list[Parameter]]:
    """Returns all parameters of the modules that should be decayed and not decayed.

    By default the following parameters are not decayed:
    - Bias
    - Normalization layers (BatchNorm, LayerNorm, GroupNorm, etc.)
    - Tokens such as cls_token, mask_token, storage_token, register_token
    - Embedding layers
    - Positional embeddings
    - 0D and 1D parameters
    - Layer scale
    - Logit scale
    - Query parameters

    Args:
        modules:
            List of modules to get the parameters from.
        extra_no_decay_keys:
            Additional parameter names that should not be decayed. Can be a substring of
            the parameter name.
        norm_layers:
            Tuple of normalization classes. Parameters from these layers will not be
            decayed.

    Returns:
        (params, params_no_weight_decay) tuple.
    """
    # Get a mapping from parameter id to (name, module, param).
    param_id_to_info: dict[int, dict[str, Any]] = {}
    for module in modules:
        for name, param in module.named_parameters():
            param_id_to_info[id(param)] = {"name": name, "param": param, "module": None}
        for mod in module.modules():
            for param in mod.parameters():
                param_id_to_info[id(param)]["module"] = mod

    params = []
    params_no_weight_decay = []

    # Iterate through each parameter and categorize it into ones that should be
    # decayed and those that should not.
    for info in param_id_to_info.values():
        param = info["param"]
        module = info["module"]
        name = info["name"]

        if (
            # Norm layers
            (isinstance(module, norm_layers) or "Norm" in module.__class__.__name__)
            # Embeddings
            or isinstance(module, (Embedding, EmbeddingBag))
            # 0D and 1D parameters
            or param.ndim <= 1
            # Biases
            or _contains_key(["bias", "bias_v", "bias_k"], name)
            # Tokens
            or _contains_key(
                [
                    "cls_token",
                    "mask_token",
                    "storage_token",
                    "register_token",
                ],
                name,
                any_pos=True,
            )
            # Queries
            or _contains_key(["queries", "query"], name, any_pos=True)
            # Positional embeddings
            or _contains_key(
                [
                    "pos_embed",
                    "position_embedding",
                    "positional_embedding",
                ],
                name,
                any_pos=True,
            )
            # Relative positional embeddings
            or _contains_key(
                [
                    "relative_position",
                    "relative_pos",
                    "rel_pos",
                    "rpb",
                    "bias_table",
                ],
                name,
                any_pos=True,
            )
            # Logit scale
            or _contains_key(
                [
                    "logit_scale",
                    "temperature",
                    "tau",
                ],
                name,
            )
            # Layer scale
            or _contains_key(
                [
                    "layer_scale",
                    "layerscale",
                    "res_scale",
                    "rezero",
                    "gamma",
                ],
                name,
            )
            # Extra user-defined keys
            or _contains_key(extra_no_decay_keys or [], name, any_pos=True)
        ):
            params_no_weight_decay.append(param)
        else:
            params.append(param)
    return params, params_no_weight_decay


def _contains_key(keys: Iterable[str], param_name: str, any_pos: bool = False) -> bool:
    if any_pos:
        return any(key in param_name for key in keys)

    # Check for whole word match in the parameter name.
    return any(
        param_name == key
        or param_name.endswith(f".{key}")
        or param_name.startswith(f"{key}.")
        or f".{key}." in param_name
        for key in keys
    )
