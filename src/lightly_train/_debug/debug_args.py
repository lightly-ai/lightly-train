#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from lightly_train._configs import validate
from lightly_train._configs.config import PydanticConfig


class DebugUnderflowOverflowArgs(PydanticConfig):
    """Arguments for underflow/overflow debugging.

    Wraps HuggingFace transformers' ``DebugUnderflowOverflow`` utility, which registers
    forward hooks on all model modules and reports the absolute min/max of every weight,
    input, and output. When a NaN or inf is detected, the last ``max_frames_to_save``
    forward frames are dumped so that the module where values first exploded can be
    identified. Training is aborted as soon as a NaN or inf is detected.
    """

    enabled: bool = False
    max_frames_to_save: int = Field(default=21, ge=1)
    trace_batch_nums: list[int] = Field(default_factory=list)
    abort_after_batch_num: int | None = Field(default=None, ge=0)

    @field_validator("trace_batch_nums")
    @classmethod
    def _validate_trace_batch_nums(cls, v: list[int]) -> list[int]:
        if any(x < 0 for x in v):
            raise ValueError("trace_batch_nums entries must be non-negative integers.")
        return v


class DebugArgs(PydanticConfig):
    underflow_overflow: DebugUnderflowOverflowArgs | None = None

    def is_underflow_overflow_enabled(self) -> bool:
        """Returns True if underflow/overflow debugging is enabled."""
        return self.underflow_overflow is not None and self.underflow_overflow.enabled


def get_debug_args(debug_args: dict[str, Any] | DebugArgs | None) -> DebugArgs:
    """Resolves the debug arguments into a :class:`DebugArgs` instance."""
    if isinstance(debug_args, DebugArgs):
        return debug_args
    debug_args = {} if debug_args is None else debug_args
    return validate.pydantic_model_validate(DebugArgs, debug_args)
