#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module


def criterion_empty_weight_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
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
