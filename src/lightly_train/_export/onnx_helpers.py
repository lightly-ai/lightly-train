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


def convert_onnx_to_float16(model_path: str) -> None:
    """Convert an ONNX model from float32 to float16.

    Uses onnxconverter-common to convert all float32 tensors (weights,
    inputs, outputs, intermediates) to float16. Integer types are left
    unchanged.

    After conversion, Cast nodes from the original graph that still
    target float32 are updated to target float16. Cast nodes inserted
    by the converter (e.g. to keep Resize inputs in float32) are left
    unchanged.
    """
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import (
        float16,  # type: ignore[import-not-found,import-untyped]
    )

    model = onnx.load(model_path)

    original_cast_names = {
        node.name for node in model.graph.node if node.op_type == "Cast"
    }

    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)

    for node in model_fp16.graph.node:
        if node.op_type == "Cast" and node.name in original_cast_names:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16

    onnx.save(model_fp16, model_path)
