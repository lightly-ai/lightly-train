#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeVar

IMAGENET_SIZE = 1_000_000  # Approximate size of the ImageNet1k train dataset


@dataclass
class ScalingInfo:
    """Information used to scale method and optimizer parameters."""

    dataset_size: int
    epochs: int


def interpolate(
    input: float,
    input_start: float,
    input_end: float,
    value_start: float,
    value_end: float,
    round_ndigits: int | None = None,
) -> float:
    """Selects a value between value_start and value_end based on the input and the
    input range."""
    value = value_start + (value_end - value_start) * (input - input_start) / (
        input_end - input_start
    )
    # Clamp the value to the range.
    value = max(value, value_start)
    value = min(value, value_end)
    return round(value, round_ndigits)


_InputType = TypeVar("_InputType", bound=float)
_ValueType = TypeVar("_ValueType")


def get_bucket_value(
    input: _InputType, buckets: Iterable[tuple[_InputType, _ValueType]]
) -> _ValueType:
    """Map an input to a value based on a set of buckets.

    Args:
        input:
            The input value that is mapped to one of the buckets.
        buckets:
            A list of (threshold, value) tuples.

    Returns:
        The value from the first bucket that has a threshold greater than the input.
    """
    for threshold, value in buckets:
        if input < threshold:
            return value
    raise ValueError(f"Input {input} is larger than all bucket thresholds.")
