#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import contextlib
from typing import Generator, Literal

import torch


def get_float32_matmul_precision(
    float32_matmul_precision: Literal["auto", "highest", "high", "medium"],
) -> Literal["highest", "high", "medium"]:
    """Get the float32 matmul precision setting."""
    if float32_matmul_precision == "auto":
        # Return torch default precision or the value set by the user if they set it
        # with torch.set_float32_matmul_precision before.
        return torch.get_float32_matmul_precision()  # type: ignore[return-value]
    else:
        return float32_matmul_precision


@contextlib.contextmanager
def float32_matmul_precision(
    float32_matmul_precision: Literal["highest", "high", "medium"],
) -> Generator[None, None, None]:
    """Context manager to temporarily set the float32 matmul precision."""
    current_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(float32_matmul_precision)
        yield
    finally:
        torch.set_float32_matmul_precision(current_precision)
