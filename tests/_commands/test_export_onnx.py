#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
import torch

from lightly_train._commands import export_onnx
from lightly_train._commands.export_onnx import ExportONNXConfig

from .. import helpers

MODEL_NAME: str = "_vit_test14"


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Fixture providing dummy input tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_backbone_weights(tmp_path: Path) -> Path:
    """Fixture providing dummy backbone weights file."""
    weights_path = tmp_path / "dummy_weights.pth"
    # Create a dummy weights file
    dummy_weights = {"dummy_key": torch.randn(10, 10)}
    torch.save(dummy_weights, weights_path)
    return weights_path


@pytest.fixture
def onnx_model_path(tmp_path: Path, dummy_backbone_weights: Path) -> Path:
    """Fixture that creates an ONNX model file for testing."""
    onnx_path: Path = tmp_path / "model.onnx"
    export_onnx.export_onnx(
        out=onnx_path,
        model_name=MODEL_NAME,
        backbone_weights=dummy_backbone_weights,
    )
    return onnx_path


def test_export_succeeds(tmp_path: Path, dummy_backbone_weights: Path) -> None:
    """Test that ONNX export succeeds and creates a valid file."""
    onnx_path: Path = tmp_path / "model.onnx"

    # Test export function
    export_onnx.export_onnx(
        out=onnx_path,
        model_name=MODEL_NAME,
        backbone_weights=dummy_backbone_weights,
    )

    # Verify file exists and has content
    assert onnx_path.exists(), "ONNX file should be created"
    assert onnx_path.stat().st_size > 0, "ONNX file should not be empty"


def test_export_with_invalid_model_name(
    tmp_path: Path, dummy_backbone_weights: Path
) -> None:
    """Test that export fails gracefully with invalid model name."""
    onnx_path: Path = tmp_path / "model.onnx"

    with pytest.raises((ValueError, KeyError, AttributeError)):
        export_onnx.export_onnx(
            out=onnx_path,
            model_name="invalid_model_name",
            backbone_weights=dummy_backbone_weights,
        )


def test_export_with_nonexistent_weights(tmp_path: Path) -> None:
    """Test that export fails gracefully with nonexistent weights file."""
    onnx_path: Path = tmp_path / "model.onnx"
    nonexistent_weights = tmp_path / "nonexistent_weights.pth"

    with pytest.raises((FileNotFoundError, RuntimeError)):
        export_onnx.export_onnx(
            out=onnx_path,
            model_name=MODEL_NAME,
            backbone_weights=nonexistent_weights,
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires Python 3.10 or higher for typing."
)
def test_export_parameters() -> None:
    """Test that export function and configs have the same parameters and default values."""
    helpers.assert_same_params(a=ExportONNXConfig, b=export_onnx.export_onnx)


def test_onnx_model_consistency_check(onnx_model_path: Path) -> None:
    """Test that the exported ONNX model passes ONNX validation."""
    onnx_model: onnx.ModelProto = onnx.load(str(onnx_model_path))

    try:
        onnx.checker.check_model(onnx_model, full_check=True)
    except Exception as e:
        pytest.fail(f"ONNX model validation failed: {e}")


def test_shape_inference_runs(onnx_model_path: Path) -> None:
    """Test that shape inference runs without errors."""
    model: onnx.ModelProto = onnx.load(str(onnx_model_path))

    try:
        inferred: onnx.ModelProto = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        pytest.fail(f"Shape inference failed: {e}")

    # Check that inference produced results
    assert inferred.graph, "Inferred model should have a graph"
    assert len(inferred.graph.input) == 1, "Inferred model should have one input"
    assert len(inferred.graph.output) == 2, "Inferred model should have two outputs"

    # Verify input has shape information
    input_shape = inferred.graph.input[0].type.tensor_type.shape
    assert input_shape.dim, "Input should have shape dimensions"


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_export_dynamic_batch(onnx_model_path: Path, batch_size: int) -> None:
    """Test that the exported model works with different batch sizes."""
    # Create input with the specified batch size
    dummy_input: torch.Tensor = torch.randn(
        batch_size, 3, 224, 224, requires_grad=False
    )

    # Create inference session
    try:
        ort_session: ort.InferenceSession = ort.InferenceSession(
            str(onnx_model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        pytest.fail(f"Failed to create ONNX Runtime session: {e}")

    # Run inference
    try:
        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(["mask", "logits"], ort_inputs)
        onnx_mask, onnx_logits = ort_outs
    except Exception as e:
        pytest.fail(f"ONNX Runtime inference failed: {e}")

    # Verify output shapes match expected batch size
    assert isinstance(onnx_mask, np.ndarray), "Mask output should be numpy array"
    assert isinstance(onnx_logits, np.ndarray), "Logits output should be numpy array"
    assert onnx_mask.shape[0] == batch_size, f"Mask batch size should be {batch_size}"
    assert onnx_logits.shape[0] == batch_size, (
        f"Logits batch size should be {batch_size}"
    )

    # Verify output shapes are consistent
    assert len(onnx_mask.shape) == 4, "Mask should have 4 dimensions (B, C, H, W)"
    assert len(onnx_logits.shape) == 4, "Logits should have 4 dimensions (B, C, H, W)"


@pytest.mark.parametrize("height,width", [(224, 224), (518, 518)])
def test_export_dynamic_height_width(
    onnx_model_path: Path, height: int, width: int
) -> None:
    """Test that the exported model works with different input dimensions."""
    # Create input with the specified height and width
    dummy_input: torch.Tensor = torch.randn(1, 3, height, width, requires_grad=False)

    # Create inference session
    try:
        ort_session: ort.InferenceSession = ort.InferenceSession(
            str(onnx_model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        pytest.fail(f"Failed to create ONNX Runtime session: {e}")

    # Run inference
    try:
        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(["mask", "logits"], ort_inputs)
        onnx_mask, onnx_logits = ort_outs
    except Exception as e:
        pytest.fail(f"ONNX Runtime inference failed with {height}x{width} input: {e}")

    # Verify output shapes match expected dimensions
    assert isinstance(onnx_mask, np.ndarray), "Mask output should be numpy array"
    assert isinstance(onnx_logits, np.ndarray), "Logits output should be numpy array"

    # Verify spatial dimensions
    assert onnx_mask.shape[0] == 1, "Mask batch size should be 1"
    assert onnx_mask.shape[2] == height, f"Mask height should be {height}"
    assert onnx_mask.shape[3] == width, f"Mask width should be {width}"
    assert onnx_logits.shape[0] == 1, "Logits batch size should be 1"
    assert onnx_logits.shape[2] == height, f"Logits height should be {height}"
    assert onnx_logits.shape[3] == width, f"Logits width should be {width}"
