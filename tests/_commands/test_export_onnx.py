#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path

import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
import torch

from ..helpers import DummyCustomModel

MODEL_NAME: str = "vits14"


@pytest.fixture
def dummy_model_eval() -> DummyCustomModel:
    """Fixture providing a dummy model for testing."""
    return DummyCustomModel().eval()


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Fixture providing dummy input tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def onnx_model_path(
    tmp_path: Path, dummy_model_eval: DummyCustomModel, dummy_input: torch.Tensor
) -> Path:
    """Fixture that creates an ONNX model file for testing."""
    onnx_path: Path = tmp_path / "model.onnx"
    torch.onnx.export(
        dummy_model_eval,
        (dummy_input,),
        str(onnx_path),
        input_names=["input"],
        output_names=["mask", "logits"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "mask": {0: "batch_size", 2: "height", 3: "width"},
            "logits": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    return onnx_path


def test_export_succeeds(
    tmp_path: Path, dummy_model_eval: DummyCustomModel, dummy_input: torch.Tensor
) -> None:
    """Test that ONNX export succeeds and creates a valid file."""
    onnx_path: Path = tmp_path / "model.onnx"
    torch.onnx.export(
        dummy_model_eval,
        (dummy_input,),
        str(onnx_path),
        input_names=["input"],
        output_names=["mask", "logits"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "mask": {0: "batch_size", 2: "height", 3: "width"},
            "logits": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


def test_onnx_model_sanity_check(onnx_model_path: Path) -> None:
    """Test that the exported ONNX model passes validation checks."""
    # Load the model
    onnx_model: onnx.ModelProto = onnx.load(str(onnx_model_path))

    # Check model structure
    onnx.checker.check_model(onnx_model, full_check=True)

    # Verify input/output names
    assert len(onnx_model.graph.input) == 1
    assert len(onnx_model.graph.output) == 2
    assert onnx_model.graph.input[0].name == "input"
    assert onnx_model.graph.output[0].name == "mask"
    assert onnx_model.graph.output[1].name == "logits"


def test_shape_inference_runs(onnx_model_path: Path) -> None:
    """Test that shape inference runs without errors."""
    model: onnx.ModelProto = onnx.load(str(onnx_model_path))
    inferred: onnx.ModelProto = onnx.shape_inference.infer_shapes(model)

    # Check that we have value info for intermediate tensors
    assert len(inferred.graph.value_info) >= 0

    # Check input/output shapes are properly inferred
    assert len(inferred.graph.input) == 1
    assert len(inferred.graph.output) == 2


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_export_dynamic_batch(onnx_model_path: Path, batch_size: int) -> None:
    """Test that the exported model works with different batch sizes."""
    # Create input with the specified batch size
    dummy_input: torch.Tensor = torch.randn(
        batch_size, 3, 224, 224, requires_grad=False
    )

    # Create inference session
    ort_session: ort.InferenceSession = ort.InferenceSession(
        str(onnx_model_path), providers=["CPUExecutionProvider"]
    )

    # Run inference
    ort_inputs = {"input": dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(["mask", "logits"], ort_inputs)
    onnx_mask, onnx_logits = ort_outs

    # Verify output shape matches expected batch size
    assert onnx_mask.shape[0] == batch_size
    assert onnx_logits.shape[0] == batch_size


@pytest.mark.parametrize("height,width", [(224, 224), (518, 518)])
def test_export_dynamic_height_width(
    onnx_model_path: Path, height: int, width: int
) -> None:
    """Test that the exported model works with different input dimensions."""
    # Create input with the specified height and width
    dummy_input: torch.Tensor = torch.randn(1, 3, height, width, requires_grad=False)

    # Create inference session
    ort_session: ort.InferenceSession = ort.InferenceSession(
        str(onnx_model_path), providers=["CPUExecutionProvider"]
    )

    # Run inference
    ort_inputs = {"input": dummy_input.cpu().numpy()}
    ort_outs = ort_session.run(["mask", "logits"], ort_inputs)
    onnx_mask, onnx_logits = ort_outs

    # Verify output shapes match expected dimensions
    assert onnx_mask.shape[0] == 1  # batch size
    assert onnx_mask.shape[2] == height
    assert onnx_mask.shape[3] == width
    assert onnx_logits.shape[0] == 1  # batch size
    assert onnx_logits.shape[2] == height
    assert onnx_logits.shape[3] == width
