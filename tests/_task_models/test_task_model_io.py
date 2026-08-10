#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections.abc import Mapping
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
    RowIndexableOutput,
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


@dataclass
class _RowOutput(RowIndexableOutput):
    scores: Tensor
    boxes: Tensor


def _row_output() -> _RowOutput:
    return _RowOutput(
        scores=torch.tensor([0.9, 0.4, 0.8]),
        boxes=torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )


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


def test_base_model_output__supports_mapping_protocol() -> None:
    output = _Output(scores=torch.ones(2), boxes=torch.zeros(2, 4))

    assert isinstance(output, Mapping)
    assert not isinstance(output, dict)

    assert set(output.keys()) == {"scores", "boxes"}
    assert len(output) == 2
    assert "scores" in output
    assert "unknown" not in output
    assert output.get("scores") is output.scores
    assert output.get("unknown", "default") == "default"

    items = dict(output.items())
    assert items["scores"] is output.scores
    assert items["boxes"] is output.boxes

    as_dict = dict(output)
    assert as_dict["scores"] is output.scores
    assert as_dict["boxes"] is output.boxes

    unpacked = {**output}
    assert unpacked["scores"] is output.scores
    assert unpacked["boxes"] is output.boxes


def test_row_indexable_output__selects_rows_with_bool_mask() -> None:
    output = _row_output()

    selected = output[output.scores > 0.5]

    assert isinstance(selected, _RowOutput)
    torch.testing.assert_close(selected.scores, torch.tensor([0.9, 0.8]))
    torch.testing.assert_close(
        selected.boxes, torch.tensor([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]])
    )


def test_row_indexable_output__selects_rows_with_index_tensor() -> None:
    output = _row_output()

    selected = output[torch.tensor([2, 0])]

    torch.testing.assert_close(selected.scores, torch.tensor([0.8, 0.9]))
    torch.testing.assert_close(
        selected.boxes, torch.tensor([[8.0, 9.0, 10.0, 11.0], [0.0, 1.0, 2.0, 3.0]])
    )


def test_row_indexable_output__selects_rows_with_list_and_slice() -> None:
    output = _row_output()

    torch.testing.assert_close(output[[1, 2]].scores, torch.tensor([0.4, 0.8]))
    torch.testing.assert_close(output[0:2].scores, torch.tensor([0.9, 0.4]))


def test_row_indexable_output__int_index_keeps_row_dimension() -> None:
    output = _row_output()

    selected = output[1]

    assert selected.boxes.shape == (1, 4)
    assert selected.scores.shape == (1,)


def test_row_indexable_output__returns_new_object_without_aliasing() -> None:
    output = _row_output()

    for selected in (output[0:2], output[output.scores > 0.0]):
        assert selected is not output
        assert selected.boxes.data_ptr() != output.boxes.data_ptr()
        selected.boxes[0, 0] = 999.0
        assert output.boxes[0, 0] == 0.0


def test_row_indexable_output__empty_selection() -> None:
    output = _row_output()

    selected = output[output.scores > 1.0]

    assert selected.scores.shape == (0,)
    assert selected.boxes.shape == (0, 4)
    assert selected.num_rows == 0


def test_row_indexable_output__num_rows_differs_from_len() -> None:
    output = _row_output()

    assert output.num_rows == 3
    assert len(output) == 2


def test_row_indexable_output__preserves_mapping_protocol() -> None:
    output = _row_output()

    assert isinstance(output, Mapping)
    assert not isinstance(output, dict)
    assert output["scores"] is output.scores
    assert set(output.keys()) == {"scores", "boxes"}
    assert "scores" in output
    assert "unknown" not in output
    assert dict(output)["boxes"] is output.boxes
    assert {**output}["scores"] is output.scores
    with pytest.raises(KeyError, match="unknown"):
        output["unknown"]

    # Mapping.__contains__ delegates to __getitem__, so a row index must not be
    # reported as a successful field lookup. Checked via __contains__ directly
    # because `0 in output` is a type error on a Mapping[str, Tensor].
    assert not output.__contains__(0)
    assert not output.__contains__(torch.tensor([0]))


@pytest.mark.parametrize(
    "index, error",
    [
        (torch.zeros(5, dtype=torch.bool), IndexError),
        (torch.zeros(2, 2, dtype=torch.bool), IndexError),
        (torch.zeros(3, dtype=torch.float32), IndexError),
        (True, TypeError),
    ],
)
def test_row_indexable_output__raises_on_invalid_index(
    index: object, error: type[Exception]
) -> None:
    output = _row_output()

    with pytest.raises(error):
        output[index]  # type: ignore[call-overload]


def test_row_indexable_output__is_registered_pytree() -> None:
    output = _row_output()

    leaves, spec = torch.utils._pytree.tree_flatten(output)
    restored = torch.utils._pytree.tree_unflatten(leaves, spec)

    assert len(leaves) == 2
    assert isinstance(restored, _RowOutput)
    torch.testing.assert_close(restored.scores, output.scores)
    torch.testing.assert_close(restored.boxes, output.boxes)


def test_to__applies_to_every_field() -> None:
    output = _Output(
        scores=torch.tensor([0.5, 0.25]), boxes=torch.ones(2, 4, dtype=torch.float32)
    )

    converted = output.to(dtype=torch.float64)

    assert isinstance(converted, _Output)
    assert converted.scores.dtype == torch.float64
    assert converted.boxes.dtype == torch.float64
    # The original is left untouched.
    assert output.scores.dtype == torch.float32


def test_to__preserves_row_indexable_type() -> None:
    output = _row_output()

    converted = output.to("cpu")

    assert isinstance(converted, _RowOutput)
    torch.testing.assert_close(converted.scores, output.scores)
    torch.testing.assert_close(converted.boxes, output.boxes)
