#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

import pytest
import torch

from lightly_train import _float32_matmul_precision


@pytest.mark.parametrize(
    "precision, expected",
    [
        ("auto", {"highest", "high", "medium"}),
        ("highest", {"highest"}),
        ("high", {"high"}),
        ("medium", {"medium"}),
    ],
)
def test_get_float32_matmul_precision__auto(
    precision: Literal["auto", "highest", "high", "medium"], expected: set[str]
) -> None:
    assert _float32_matmul_precision.get_float32_matmul_precision(precision) in expected


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_float32_matmul_precision(
    precision: Literal["highest", "high", "medium"],
) -> None:
    default = torch.get_float32_matmul_precision()
    with _float32_matmul_precision.float32_matmul_precision(precision):
        assert torch.get_float32_matmul_precision() == precision
    assert torch.get_float32_matmul_precision() == default
