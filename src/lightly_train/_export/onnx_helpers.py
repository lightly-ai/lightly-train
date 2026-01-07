#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from enum import Enum

import torch

_PRECALCULATE_FOR_ONNX_EXPORT = contextvars.ContextVar(
    "PRECALCULATE_FOR_ONNX_EXPORT", default=False
)


def is_in_precalculate_for_onnx_export() -> bool:
    return _PRECALCULATE_FOR_ONNX_EXPORT.get()


@contextlib.contextmanager
def precalculate_for_onnx_export() -> Iterator[None]:
    """
    For certain models we want to precalculate some values and store them in the model
    before exporting the model to ONNX. In order to avoid having to pass those options
    through all methods we have this context manager. Therefore, one should call
    ```
    with precalculate_for_onnx_export():
        model(example_input)
    ```
    before running `torch.onnx.export(model, example_input)`.
    In the relevant part of the model we can check if we are in this context with
    `is_in_precalculate_for_onnx_export()`.
    """
    token = _PRECALCULATE_FOR_ONNX_EXPORT.set(True)
    try:
        yield
    finally:
        _PRECALCULATE_FOR_ONNX_EXPORT.reset(token)


class ONNXPrecision(str, Enum):
    F16_TRUE = "16-true"
    F32_TRUE = "32-true"

    def torch_dtype(self) -> torch.dtype:
        return {
            ONNXPrecision.F32_TRUE: torch.float32,
            ONNXPrecision.F16_TRUE: torch.float16,
        }[self]
