#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.export.dynamic_shapes import Dim, _DimHint
from torch.nn import Module

from lightly_train._task_models import task_model as task_model_module
from lightly_train._task_models.task_model import (
    ExportMixin,
    ONNXExportPrecisionPolicy,
    TaskModel,
)
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)

_DYNAMIC_DIM: Any = _DimHint.DYNAMIC()  # type: ignore[no-untyped-call]
_STATIC_DIM: Any = _DimHint.STATIC()  # type: ignore[no-untyped-call]


@dataclass
class _Output(BaseModelOutput):
    scores: Tensor


class _ExportModel(TaskModel, ExportMixin):
    def __init__(self) -> None:
        super().__init__(init_args={"self": self, "__class__": self.__class__})
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.proj = torch.nn.Linear(1, 1)
        self.events: list[str] = []
        self.verified_outputs: tuple[BaseModelOutput, BaseModelOutput] | None = None

    @property
    def model_input_spec(self) -> ModelInputSpec:
        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, 8, 8),
                    dtype=torch.float32,
                    is_batched=True,
                )
            },
            input_dynamic_shapes={
                "images": (
                    _DYNAMIC_DIM,
                    _STATIC_DIM,
                    _STATIC_DIM,
                    _STATIC_DIM,
                )
            },
        )

    def deploy(self) -> "_ExportModel":
        self.events.append("deploy")
        return self

    def forward(self, images: Tensor) -> _Output:
        return _Output(scores=images.mean(dim=(1, 2, 3)) * self.weight)

    def verify_onnx_export_outputs(
        self,
        *,
        torch_outputs: BaseModelOutput,
        onnx_outputs: BaseModelOutput,
    ) -> None:
        self.verified_outputs = (torch_outputs, onnx_outputs)


def test_export_mixin__does_not_compile_forward(monkeypatch: Any) -> None:
    calls = []

    def compile_spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return args[0]

    monkeypatch.setattr(torch, "compile", compile_spy)

    class Model(_ExportModel):
        pass

    Model()

    assert calls == []


def test_tensor_spec__example_tensor_applies_batch_size() -> None:
    spec = TensorSpec(shape=(3, 8, 8), dtype=torch.float32, is_batched=True)

    tensor = spec.example_tensor(batch_size=2)

    assert tensor.shape == (2, 3, 8, 8)
    assert tensor.dtype == torch.float32


def test_model_input_spec__example_inputs_uses_declared_specs() -> None:
    spec = ModelInputSpec(
        input_specs={
            "images": TensorSpec(
                shape=(3, 8, 8),
                dtype=torch.float32,
                is_batched=True,
            ),
            "indices": TensorSpec(
                shape=(4,),
                dtype=torch.int64,
                is_batched=False,
            ),
        },
        input_dynamic_shapes={
            "images": (_DYNAMIC_DIM, _STATIC_DIM, _STATIC_DIM, _STATIC_DIM),
            "indices": (_STATIC_DIM,),
        },
    )

    inputs = spec.example_inputs()

    assert inputs["images"].shape == (1, 3, 8, 8)
    assert inputs["images"].dtype == torch.float32
    assert inputs["indices"].shape == (4,)
    assert inputs["indices"].dtype == torch.int64


def test_model_input_spec__example_inputs_uses_dynamic_batch_min() -> None:
    spec = ModelInputSpec(
        input_specs={
            "images": TensorSpec(
                shape=(3, 8, 8),
                dtype=torch.float32,
                is_batched=True,
            ),
        },
        input_dynamic_shapes={
            "images": (
                Dim("batch_size", min=4, max=8),
                _STATIC_DIM,
                _STATIC_DIM,
                _STATIC_DIM,
            ),
        },
    )

    inputs = spec.example_inputs()

    assert inputs["images"].shape == (4, 3, 8, 8)


def test_model_input_spec__example_inputs_batch_size_overrides_default() -> None:
    spec = ModelInputSpec(
        input_specs={
            "images": TensorSpec(
                shape=(3, 8, 8),
                dtype=torch.float32,
                is_batched=True,
            ),
        },
        input_dynamic_shapes={
            "images": (
                Dim("batch_size", min=4, max=8),
                _STATIC_DIM,
                _STATIC_DIM,
                _STATIC_DIM,
            ),
        },
    )

    inputs = spec.example_inputs(batch_size=2)

    assert inputs["images"].shape == (2, 3, 8, 8)


def test_model_input_spec__rejects_dynamic_non_batch_dimension() -> None:
    with pytest.raises(ValueError, match="Only the batch dimension may be dynamic"):
        ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, 8, 8),
                    dtype=torch.float32,
                    is_batched=True,
                ),
            },
            input_dynamic_shapes={
                "images": (
                    _DYNAMIC_DIM,
                    _STATIC_DIM,
                    _DYNAMIC_DIM,
                    _STATIC_DIM,
                ),
            },
        )


def test_model_input_spec__rejects_dynamic_unbatched_dimension() -> None:
    with pytest.raises(ValueError, match="Only the batch dimension may be dynamic"):
        ModelInputSpec(
            input_specs={
                "indices": TensorSpec(
                    shape=(4,),
                    dtype=torch.int64,
                    is_batched=False,
                ),
            },
            input_dynamic_shapes={
                "indices": (_DYNAMIC_DIM,),
            },
        )


def test_export_onnx__uses_model_input_spec_names_and_deploy(
    tmp_path: Path, monkeypatch: Any
) -> None:
    record: dict[str, Any] = {}

    monkeypatch.setattr(
        task_model_module, "check_onnx_dynamo_requirements", lambda: None
    )

    def export_spy(
        module: Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Tensor],
        f: str,
        input_names: list[str],
        output_names: list[str],
        opset_version: int | None,
        dynamo: bool,
        dynamic_shapes: dict[str, tuple[Any, ...]],
        **format_args: Any,
    ) -> None:
        record.update(
            args=args,
            kwargs=kwargs,
            input_names=input_names,
            output_names=output_names,
            dynamo=dynamo,
            dynamic_shapes=dynamic_shapes,
            format_args=format_args,
        )

    monkeypatch.setattr(torch.onnx, "export", export_spy)

    model = _ExportModel()
    model.export_onnx(
        tmp_path / "model.onnx",
        height=10,
        width=12,
        simplify=False,
        verify=False,
        format_args={"fallback": True},
    )

    assert model.events == ["deploy"]
    assert record["args"] == ()
    assert list(record["kwargs"]) == ["images"]
    assert record["kwargs"]["images"].shape == (2, 3, 10, 12)
    assert record["input_names"] == ["images"]
    assert record["output_names"] == ["scores"]
    assert record["dynamo"] is True
    assert record["dynamic_shapes"]["images"][0] == _DYNAMIC_DIM
    assert record["format_args"] == {"fallback": True}


def test_export_onnx__module_precision_policy(tmp_path: Path, monkeypatch: Any) -> None:
    record: dict[str, Any] = {}

    monkeypatch.setattr(
        task_model_module, "check_onnx_dynamo_requirements", lambda: None
    )

    class PrecisionModel(_ExportModel):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Sequential(torch.nn.Linear(1, 1))

        @property
        def onnx_export_precision_policy(self) -> ONNXExportPrecisionPolicy:
            return ONNXExportPrecisionPolicy(fp32_module_names=("block",))

    def export_spy(
        module: Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Tensor],
        f: str,
        input_names: list[str],
        output_names: list[str],
        opset_version: int | None,
        dynamo: bool,
        dynamic_shapes: dict[str, tuple[Any, ...]],
        **format_args: Any,
    ) -> None:
        assert isinstance(module, PrecisionModel)
        record["input_dtype"] = kwargs["images"].dtype
        record["weight_dtype"] = module.weight.dtype
        record["proj_dtype"] = module.proj.weight.dtype
        record["block_child_dtype"] = module.block[0].weight.dtype

    monkeypatch.setattr(torch.onnx, "export", export_spy)

    model = PrecisionModel()
    model.export_onnx(
        tmp_path / "model.onnx",
        precision="fp16",
        simplify=False,
        verify=False,
    )

    assert record == {
        "input_dtype": torch.float16,
        "weight_dtype": torch.float16,
        "proj_dtype": torch.float16,
        "block_child_dtype": torch.float32,
    }


def test_export_onnx__graph_precision_policy(tmp_path: Path, monkeypatch: Any) -> None:
    calls: list[tuple[str, Any]] = []
    fake_onnx = ModuleType("onnx")
    fake_float16 = ModuleType("float16")
    fake_transformers = ModuleType("onnxruntime.transformers")
    fake_runtime = ModuleType("onnxruntime")

    def load(path: str) -> str:
        calls.append(("load", path))
        return "model"

    def save(model: str, path: str) -> None:
        calls.append(("save", (model, path)))

    def convert_float_to_float16(
        model: str,
        *,
        op_block_list: list[str],
    ) -> str:
        calls.append(("convert", op_block_list))
        return "model-fp16"

    fake_onnx.load = load  # type: ignore[attr-defined]
    fake_onnx.save = save  # type: ignore[attr-defined]
    fake_float16.DEFAULT_OP_BLOCK_LIST = ["LayerNormalization"]  # type: ignore[attr-defined]
    fake_float16.convert_float_to_float16 = convert_float_to_float16  # type: ignore[attr-defined]
    fake_transformers.float16 = fake_float16  # type: ignore[attr-defined]
    fake_runtime.transformers = fake_transformers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.transformers",
        fake_transformers,
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime.transformers.float16",
        fake_float16,
    )
    monkeypatch.setattr(
        task_model_module,
        "remove_redundant_casts",
        lambda model: calls.append(("remove_casts", model)),
    )
    monkeypatch.setattr(
        task_model_module,
        "fix_topological_order",
        lambda model: calls.append(("fix_order", model)),
    )

    class PrecisionModel(_ExportModel):
        @property
        def onnx_export_precision_policy(self) -> ONNXExportPrecisionPolicy:
            return ONNXExportPrecisionPolicy(fp32_onnx_op_types=("Softmax", "MatMul"))

    model = PrecisionModel()
    out = tmp_path / "model.onnx"
    model._apply_onnx_export_graph_precision(out=out, precision="fp16")

    assert calls == [
        ("load", str(out)),
        ("convert", ["LayerNormalization", "Softmax", "MatMul"]),
        ("remove_casts", "model-fp16"),
        ("fix_order", "model-fp16"),
        ("save", ("model-fp16", str(out))),
    ]


def test_verify_onnx_export__reconstructs_base_model_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    out = tmp_path / "model.onnx"
    out.touch()
    fake_onnx = ModuleType("onnx")
    fake_ort = ModuleType("onnxruntime")
    fake_checker = SimpleNamespace(
        check_model=lambda path, full_check: None,
    )

    class FakeSession:
        def __init__(self, path: str) -> None:
            self.path = path

        def run(
            self,
            output_names: None,
            input_feed: dict[str, Any],
        ) -> list[Any]:
            return [input_feed["images"].mean(axis=(1, 2, 3))]

    fake_onnx.checker = fake_checker  # type: ignore[attr-defined]
    fake_ort.InferenceSession = FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model = _ExportModel()
    example_inputs = {
        "images": torch.ones(2, 3, 8, 8),
    }

    model._verify_onnx_export(
        out=out,
        module=model,
        example_inputs=example_inputs,
    )

    assert model.verified_outputs is not None
    torch_outputs, onnx_outputs = model.verified_outputs
    assert isinstance(torch_outputs, _Output)
    assert isinstance(onnx_outputs, _Output)
    assert torch_outputs.scores.shape == (2,)
    assert onnx_outputs.scores.shape == (2,)
