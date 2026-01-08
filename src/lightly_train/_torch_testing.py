#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Callable

from torch import Tensor


def assert_most_equal(
    actual: Tensor,
    expected: Tensor,
    min_fraction: float = 0.99,
    msg: str | Callable[[str], str] | None = None,
) -> None:
    """Asserts that at least `min_fraction` of the elements in tensors `actual` and
    `expected` are equal.

    This is useful for verifying that two model outputs are similar when the output
    types are bool/int/long. For example segmentation masks.

    Use torch.testing.assert_close for float tensors like model logits.
    """
    err_msg: str | None = None
    if actual.shape != expected.shape:
        err_msg = f"shape mismatch: {actual.shape} vs {expected.shape}"

    if err_msg is None:
        eq = actual == expected
        frac = eq.float().mean().item()
        if frac < min_fraction:
            err_msg = f"only {frac:.4f} of {actual.numel()} elements are equal"

    if err_msg is not None:
        if msg is None:
            raise AssertionError(err_msg)
        elif isinstance(msg, str):
            raise AssertionError(msg)
        elif callable(msg):
            raise AssertionError(msg(err_msg))
        else:
            raise TypeError(f"msg must be str or callable, got {type(msg)}")
