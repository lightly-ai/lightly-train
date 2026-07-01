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

    enabled: bool = Field(
        default=False,
        description=(
            "Whether to enable underflow/overflow debugging. When True, registers "
            "forward hooks on all model modules to detect inf/nan in activations and "
            "weights. Increases training time significantly — disable after debugging."
        ),
    )
    max_frames_to_save: int = Field(
        default=21,
        ge=1,
        description=(
            "How many forward-pass frames to retain when dumping context after an "
            "inf/nan is detected. The most recent N frames are written to the debug "
            "log so the module where values first exploded can be identified."
        ),
    )
    trace_batch_nums: list[int] = Field(
        default_factory=list,
        description=(
            "Training-step numbers at which to write a full absolute min/max trace "
            "of every weight, input and output to the debug log. Detection is "
            "disabled on traced steps (training does not abort). Useful for "
            "fast-forwarding to a known-bad region. Step numbers are 0-indexed and "
            "must be non-negative."
        ),
    )
    abort_after_batch_num: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional training-step after which to abort. When set, training raises "
            "ValueError once the current step exceeds this threshold. Mainly useful "
            "in combination with `trace_batch_nums` to inspect a region and stop."
        ),
    )

    @field_validator("trace_batch_nums")
    @classmethod
    def _validate_trace_batch_nums(cls, v: list[int]) -> list[int]:
        if any(x < 0 for x in v):
            raise ValueError("trace_batch_nums entries must be non-negative integers.")
        return v


class NaNCaptureArgs(PydanticConfig):
    """Arguments for NaN/Inf capture debugging.

    When enabled, the monitor scans parameter gradients for NaN/Inf after each
    gradient-accumulation step (before ``clip_gradients``/``optimizer.step``).
    On detection, it saves a self-contained capture to
    ``out_dir/debug/nan_capture/rank{R}/nan_capture.pt`` holding the model
    state, the step's microbatches, RNG state, and metadata — then raises
    :class:`NaNDetectedError` to halt training. The capture is reproducibly
    loadable via :func:`lightly_train._debug.nan_capture.load_nan_capture`
    and ``.replay()``.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Whether to enable NaN/Inf capture debugging. When True, scans "
            "parameter gradients after each accumulated training step and, on "
            "first non-finite gradient, saves a replayable capture to "
            "out_dir/debug/nan_capture/rank{R}/nan_capture.pt before aborting."
        ),
    )


class DebugArgs(PydanticConfig):
    underflow_overflow: DebugUnderflowOverflowArgs | None = None
    nancapture: NaNCaptureArgs | None = None

    def is_underflow_overflow_enabled(self) -> bool:
        """Returns True if underflow/overflow debugging is enabled."""
        return self.underflow_overflow is not None and self.underflow_overflow.enabled

    def is_nancapture_enabled(self) -> bool:
        """Returns True if NaN/Inf capture debugging is enabled."""
        return self.nancapture is not None and self.nancapture.enabled


def get_debug_args(debug_args: dict[str, Any] | DebugArgs | None) -> DebugArgs:
    """Resolves the debug arguments into a :class:`DebugArgs` instance."""
    if isinstance(debug_args, DebugArgs):
        return debug_args
    debug_args = {} if debug_args is None else debug_args
    return validate.pydantic_model_validate(DebugArgs, debug_args)
