#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._task_models.object_detection_components.ltdetr_schedule import (
    LTDETRStepSchedule,
    resolve_ltdetr_step_schedule,
)


@pytest.mark.parametrize(
    ("total_steps", "step_start", "step_flat", "step_stop"),
    [
        (100, 0, 0, 100),
        (200, 0, 100, 200),
        (300, 100, 200, 300),
        (400, 100, 300, 300),
        (1200, 400, 1000, 1000),
        (7200, 400, 4000, 6000),
    ],
)
def test_resolve_ltdetr_step_schedule__resolved_windows(
    total_steps: int,
    step_start: int,
    step_flat: int,
    step_stop: int,
) -> None:
    schedule = resolve_ltdetr_step_schedule(
        total_steps=total_steps,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )
    assert schedule == LTDETRStepSchedule(
        step_start=step_start,
        step_flat=step_flat,
        step_stop=step_stop,
    )
