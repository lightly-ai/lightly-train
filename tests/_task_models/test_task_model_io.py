#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.export import Dim

from lightly_train._export.onnx_helpers import (
    _TORCH_DIM_HINTS_AVAILABLE,
    _TORCH_DIM_HINTS_MIN_VERSION,
)
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)

requires_dim_hints = pytest.mark.skipif(
    not _TORCH_DIM_HINTS_AVAILABLE,
    reason=f"torch >= {_TORCH_DIM_HINTS_MIN_VERSION} required",
)


@dataclass
class _Output(BaseModelOutput):
    scores: Tensor
    boxes: Tensor


def test_tensor_spec__example_tensor() -> None:
    spec = TensorSpec(shape=(3, 8, 8), dtype=torch.float32, is_batched=True)
    tensor = spec.example_tensor(batch_size=2)
    assert tensor.shape == (2, 3, 8, 8)
    assert tensor.dtype == torch.float32


@requires_dim_hints
def test_model_input_spec__example_inputs_and_dynamic_batch() -> None:
    spec = ModelInputSpec(
        input_specs={
            "images": TensorSpec(shape=(3, 8, 8), dtype=torch.float32, is_batched=True)
        },
        input_dynamic_shapes={
            "images": (Dim("batch", min=4), Dim.STATIC, Dim.STATIC, Dim.STATIC)
        },
    )
    assert spec.example_inputs()["images"].shape == (4, 3, 8, 8)
    assert spec.example_inputs(batch_size=2)["images"].shape == (2, 3, 8, 8)
    assert spec.dynamic_shapes(dynamic_batch_size=False)["images"][0] == Dim.STATIC


@requires_dim_hints
def test_model_input_spec__example_inputs_shape_overrides() -> None:
    spec = ModelInputSpec(
        input_specs={
            "images": TensorSpec(shape=(3, 8, 8), dtype=torch.float32, is_batched=True)
        },
        input_dynamic_shapes={
            "images": (Dim("batch", min=1), Dim.STATIC, Dim.STATIC, Dim.STATIC)
        },
    )

    inputs = spec.example_inputs(
        batch_size=2, shape_overrides={"images": (1, None, 12)}
    )

    assert inputs["images"].shape == (2, 1, 8, 12)
    assert spec.input_specs["images"].shape == (3, 8, 8)


@requires_dim_hints
def test_model_input_spec__rejects_dynamic_non_batch_dimension() -> None:
    with pytest.raises(ValueError, match="Only the batch dimension may be dynamic"):
        ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, 8, 8), dtype=torch.float32, is_batched=True
                )
            },
            input_dynamic_shapes={
                "images": (Dim.DYNAMIC, Dim.STATIC, Dim.DYNAMIC, Dim.STATIC)
            },
        )


def test_base_model_output__is_registered_pytree() -> None:
    output = _Output(scores=torch.ones(2), boxes=torch.zeros(2, 4))
    values, context = torch.utils._pytree.tree_flatten(output)
    assert context is not None
    assert len(values) == 2
    restored = torch.utils._pytree.tree_unflatten(values, context)
    assert isinstance(restored, _Output)
    torch.testing.assert_close(restored.scores, output.scores)
    torch.testing.assert_close(restored.boxes, output.boxes)


def test_base_model_output__supports_keyed_access() -> None:
    output = _Output(scores=torch.ones(2), boxes=torch.zeros(2, 4))

    assert output["scores"] is output.scores
    assert output["boxes"] is output.boxes
    with pytest.raises(KeyError, match="unknown"):
        output["unknown"]
