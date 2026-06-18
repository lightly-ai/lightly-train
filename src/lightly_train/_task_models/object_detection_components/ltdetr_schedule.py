#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from dataclasses import dataclass

# Matched upstream schedule profile for LTDETR.
LTDETR_REFERENCE_WARMUP_EPOCHS = 4
LTDETR_SHORT_RUN_TOTAL_EPOCHS = 12
LTDETR_REFERENCE_TOTAL_EPOCHS = 72
LTDETR_REFERENCE_NO_AUG_EPOCHS = 12


@dataclass(frozen=True)
class LTDETRStepSchedule:
    step_start: int
    step_flat: int
    step_stop: int


def resolve_ltdetr_step_schedule(
    *,
    total_steps: int,
    train_num_batches: int,
    gradient_accumulation_steps: int,
) -> LTDETRStepSchedule:
    """Resolve ``"auto"`` for LTDETR step schedule.

    The algorithm converts LTDETR's epoch-based schedule into concrete step
    boundaries using the effective number of optimizer steps per epoch,
    ``train_num_batches / gradient_accumulation_steps``.

    The calculation works in four stages:

    1. Derive the effective training length in epochs as
       ``total_steps / steps_per_epoch``.
    2. Scale the canonical LTDETR no-augmentation tail from the matched
       upstream profile:

           no_aug_epochs_resolved = min(
               _REFERENCE_NO_AUG_EPOCHS,
               round(
                   total_epochs
                   * _REFERENCE_NO_AUG_EPOCHS
                   / _REFERENCE_TOTAL_EPOCHS
               ),
           )

    3. Convert that recipe into three epoch boundaries:
       - ``epoch_stop_resolved = total_epochs - no_aug_epochs_resolved``
       - short runs with ``total_epochs <= LTDETR_SHORT_RUN_TOTAL_EPOCHS`` compress
         the warmup to
         ``floor(total_epochs / (LTDETR_SHORT_RUN_TOTAL_EPOCHS / LTDETR_REFERENCE_WARMUP_EPOCHS))``
         and set ``epoch_flat_resolved`` to
         ``min(epoch_stop_resolved, epoch_start_resolved + floor(total_epochs / 2))``
       - longer runs keep ``epoch_start_resolved = LTDETR_REFERENCE_WARMUP_EPOCHS`` and use
         ``epoch_flat_resolved = LTDETR_REFERENCE_WARMUP_EPOCHS + floor(total_epochs / 2)``
    4. Convert each epoch boundary back to integer steps with
       ``floor(epoch * steps_per_epoch)``.

    The returned schedule contains three step boundaries:
    - ``step_start`` begins the scheduled LTDETR transition window.
    - ``step_flat`` marks the boundary between the flat phase and the
      no-augmentation tail.
    - ``step_stop`` begins the no-augmentation tail.
    """

    steps_per_epoch = train_num_batches / gradient_accumulation_steps
    total_epochs = total_steps / steps_per_epoch

    no_aug_epochs_resolved = min(
        LTDETR_REFERENCE_NO_AUG_EPOCHS,
        round(
            total_epochs
            * LTDETR_REFERENCE_NO_AUG_EPOCHS
            / LTDETR_REFERENCE_TOTAL_EPOCHS
        ),
    )
    epoch_stop_resolved = total_epochs - no_aug_epochs_resolved

    if total_epochs <= LTDETR_SHORT_RUN_TOTAL_EPOCHS:
        epoch_start_resolved = math.floor(
            total_epochs
            / (LTDETR_SHORT_RUN_TOTAL_EPOCHS / LTDETR_REFERENCE_WARMUP_EPOCHS)
        )
        epoch_flat_resolved = min(
            epoch_stop_resolved,
            epoch_start_resolved + math.floor(total_epochs / 2),
        )
    else:
        epoch_start_resolved = LTDETR_REFERENCE_WARMUP_EPOCHS
        epoch_flat_resolved = LTDETR_REFERENCE_WARMUP_EPOCHS + math.floor(
            total_epochs / 2
        )

    return LTDETRStepSchedule(
        step_start=math.floor(epoch_start_resolved * steps_per_epoch),
        step_flat=math.floor(epoch_flat_resolved * steps_per_epoch),
        step_stop=math.floor(epoch_stop_resolved * steps_per_epoch),
    )
