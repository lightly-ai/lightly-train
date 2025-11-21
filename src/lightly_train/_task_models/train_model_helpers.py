#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any

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

    del state_dict[criterion_empty_weight_key]
