#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math

_REFERENCE_TOTAL_EPOCHS = 74
_REFERENCE_NO_AUG_EPOCHS = 2


def resolve_no_aug_steps(
    *,
    total_steps: int,
    train_num_batches: int,
    gradient_accumulation_steps: int,
) -> int:
    """Resolve the no-augmentation tail for the ECSeg schedule."""
    steps_per_epoch = train_num_batches / gradient_accumulation_steps
    total_epochs = total_steps / steps_per_epoch
    no_aug_epochs = min(
        _REFERENCE_NO_AUG_EPOCHS,
        round(total_epochs * _REFERENCE_NO_AUG_EPOCHS / _REFERENCE_TOTAL_EPOCHS),
    )
    return math.floor(no_aug_epochs * steps_per_epoch)
