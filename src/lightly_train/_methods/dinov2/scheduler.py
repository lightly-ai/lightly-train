#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# References:
# - https://github.com/lightly-ai/lightly/blob/master/lightly/utils/scheduler.py


def linear_warmup_schedule(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
) -> float:  # TODO: import from LightlySSL after new release
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if start_value < 0:
        raise ValueError(f"Start value {start_value} can't be negative.")
    if end_value <= 0:
        raise ValueError(f"End value {end_value} can't be non-positive.")
    if start_value > end_value:
        raise ValueError(
            f"Start value {start_value} must be less than or equal to end value {end_value}."
        )
    if step < warmup_steps:
        return start_value + step / warmup_steps * (end_value - start_value)
    else:
        return end_value
