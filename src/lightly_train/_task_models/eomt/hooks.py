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
from torch.nn import Module

logger = logging.getLogger(__name__)


def queries_adjust_num_queries_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Resize query embeddings from the checkpoint to match the module configuration."""
    queries_weight_key = f"{prefix}queries.weight"
    queries_weight = state_dict.get(queries_weight_key)
    if queries_weight is None:
        return

    query_embed_module = getattr(module, "queries", None)
    num_queries_module = getattr(module, "num_queries", None)
    if query_embed_module is None or num_queries_module is None:
        return

    num_queries_state = queries_weight.shape[0]
    if num_queries_state == num_queries_module:
        return
    elif num_queries_state > num_queries_module:
        logger.info(
            f"Checkpoint provides {num_queries_state} queries but module expects {num_queries_module}. Truncating.",
        )

        queries_weight = queries_weight[:num_queries_module, :]
    else:
        logger.info(
            f"Checkpoint provides {num_queries_state} queries but module expects {num_queries_module}. Repeating entries.",
        )

        repeated_times, remainder = divmod(num_queries_module, num_queries_state)
        queries_weight = queries_weight.repeat(repeated_times, 1)
        if remainder > 0:
            queries_weight = torch.cat(
                [queries_weight, queries_weight[:remainder, :]], dim=0
            )

    state_dict[queries_weight_key] = queries_weight


def class_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reuse or reinitialize class head when number of classes changes."""
    class_head_weight_key = f"{prefix}class_head.weight"
    class_head_bias_key = f"{prefix}class_head.bias"
    class_head_weight = state_dict.get(class_head_weight_key)
    if class_head_weight is None:
        return

    class_head_module = getattr(module, "class_head", None)
    if class_head_module is None:
        return

    num_classes_state = class_head_weight.shape[0]
    num_classes_module = class_head_module.out_features
    if num_classes_state == num_classes_module:
        return
    else:
        logger.info(
            f"Checkpoint provides {num_classes_state - 1} classes but module expects {num_classes_module - 1}. Reinitializing class head.",
        )

        # Keep the module initialization by overwriting the checkpoint weights with the
        # current parameter tensors.
        state_dict[class_head_weight_key] = class_head_module.weight.detach().clone()
        state_dict[class_head_bias_key] = class_head_module.bias.detach().clone()


def criterion_empty_weight_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reinitialize criterion empty weight buffer to match model configuration."""
    criterion_empty_weight_key = f"{prefix}criterion.empty_weight"
    criterion_empty_weight = state_dict.get(criterion_empty_weight_key)
    if criterion_empty_weight is None:
        return

    criterion_module = getattr(module, "criterion", None)
    if criterion_module is None:
        return

    model_args_module = getattr(module, "model_args", None)
    if model_args_module is None:
        return

    # Re-initialize the empty weight buffer to match the current
    criterion_empty_weight_reinit = torch.ones_like(criterion_module.empty_weight)
    criterion_empty_weight_reinit[-1] = model_args_module.loss_no_object_coefficient

    state_dict[criterion_empty_weight_key] = criterion_empty_weight_reinit
