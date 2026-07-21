#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from torch.export import ExportedProgram

from lightly_train._export.export_onnx import ONNXExportMixin
from lightly_train._export.onnx_helpers import (
    _TORCH_DIM_HINTS_AVAILABLE,
    _TORCH_DIM_HINTS_MIN_VERSION,
    _TORCH_DYNAMO_AVAILABLE,
    _TORCH_DYNAMO_MIN_VERSION,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)


@dataclass
class _Output(BaseModelOutput):
    scores: Tensor


class _Model(TaskModel, ONNXExportMixin):
    def __init__(self) -> None:
        super().__init__(init_args=locals())
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.deployed = False

    @property
    def model_input_spec(self) -> ModelInputSpec:
        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, 8, 8), dtype=torch.float32, is_batched=True
                )
            },
            input_dynamic_shapes={
                "images": (
                    torch.export.Dim("batch", min=1, max=8),
                    torch.export.Dim.STATIC,
                    torch.export.Dim.STATIC,
                    torch.export.Dim.STATIC,
                )
            },
        )

    def deploy(self) -> _Model:
        self.deployed = True
        return self

    def forward(self, images: Tensor) -> _Output:
        return _Output(scores=images.mean(dim=(1, 2, 3)) * self.weight)

    def verify_onnx_export_outputs(
        self, *, torch_outputs: BaseModelOutput, onnx_outputs: BaseModelOutput
    ) -> None:
        assert isinstance(torch_outputs, _Output)
        assert isinstance(onnx_outputs, _Output)
        torch.testing.assert_close(torch_outputs.scores, onnx_outputs.scores)


@pytest.mark.skipif(
    not (_TORCH_DYNAMO_AVAILABLE and _TORCH_DIM_HINTS_AVAILABLE),
    reason=(
        f"torch >= {_TORCH_DYNAMO_MIN_VERSION} (dynamo export) and "
        f"torch >= {_TORCH_DIM_HINTS_MIN_VERSION} (Dim hints) required"
    ),
)
def test_export_mixin__creates_strict_exported_program() -> None:
    model = _Model()

    program = model.export()

    assert isinstance(program, ExportedProgram)
    assert program.example_inputs[1]["images"].shape == (2, 3, 8, 8)


def test_onnx_export_mixin__rejects_invalid_precision(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = _Model()

    with pytest.raises(
        ValueError,
        match="Invalid precision 'bf16'. Must be one of 'fp32', 'fp16'.",
    ):
        model.export_onnx(
            tmp_path / "model.onnx",
            precision="bf16",  # type: ignore[arg-type]
        )

    assert not model.deployed


@pytest.mark.skipif(
    not (_TORCH_DYNAMO_AVAILABLE and _TORCH_DIM_HINTS_AVAILABLE),
    reason=(
        f"torch >= {_TORCH_DYNAMO_MIN_VERSION} (dynamo export) and "
        f"torch >= {_TORCH_DIM_HINTS_MIN_VERSION} (Dim hints) required"
    ),
)
def test_onnx_export_mixin__deploys_model_and_embeds_metadata(tmp_path) -> None:  # type: ignore[no-untyped-def]
    onnx = pytest.importorskip("onnx")

    model = _Model()
    out = tmp_path / "model.onnx"

    model.export_onnx(out, simplify=False, verify=False)

    assert model.deployed
    model_onnx = onnx.load(str(out))
    assert [input_.name for input_ in model_onnx.graph.input] == ["images"]
    assert [output.name for output in model_onnx.graph.output] == ["scores"]
    metadata = {entry.key: entry.value for entry in model_onnx.metadata_props}
    assert "lightly_train_version" in metadata
