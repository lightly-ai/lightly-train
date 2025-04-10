#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train import _scaling


@pytest.mark.parametrize(
    "input, input_start, input_end, value_start, value_end, expected",
    [
        (0, 0, 1, 10, 20, 10),
        (0.5, 0, 1, 10, 20, 15),
        (1, 0, 1, 10, 20, 20),
        (-1, 0, 1, 10, 20, 10),  # clamp to lower value
        (2, 0, 1, 10, 20, 20),  # clamp to upper value
    ],
)
def test_interpolate(
    input: float,
    input_start: float,
    input_end: float,
    value_start: float,
    value_end: float,
    expected: float,
) -> None:
    assert (
        _scaling.interpolate(
            input=input,
            input_start=input_start,
            input_end=input_end,
            value_start=value_start,
            value_end=value_end,
        )
        == expected
    )


@pytest.mark.parametrize(
    "input, buckets, expected",
    [
        (0, [(1, 10), (2, 20)], 10),
        (1, [(1, 10), (2, 20)], 20),
    ],
)
def test_get_bucket_value(
    input: int, buckets: list[tuple[int, int]], expected: int
) -> None:
    assert _scaling.get_bucket_value(input=input, buckets=buckets) == expected


def test_get_bucket_value__input_too_large() -> None:
    with pytest.raises(ValueError):
        _scaling.get_bucket_value(input=2, buckets=[(1, 10), (2, 20)])
