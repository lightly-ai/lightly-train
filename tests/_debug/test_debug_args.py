#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._debug.debug_args import get_debug_args
from lightly_train.errors import ConfigValidationError


def test_get_debug_args__parses_user_config() -> None:
    assert get_debug_args(None).is_underflow_overflow_enabled() is False

    args = get_debug_args(
        {"underflow_overflow": {"enabled": True, "max_frames_to_save": 5}}
    )

    assert args.is_underflow_overflow_enabled() is True
    assert args.underflow_overflow is not None
    assert args.underflow_overflow.max_frames_to_save == 5


def test_get_debug_args__rejects_unknown_keys() -> None:
    with pytest.raises(ConfigValidationError):
        get_debug_args({"underflow_overflow": {"nonexistent_key": 1}})


def test_get_debug_args__trace_batch_nums_rejects_negative() -> None:
    with pytest.raises(ConfigValidationError, match="non-negative"):
        get_debug_args({"underflow_overflow": {"trace_batch_nums": [-1]}})
