#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import pytest
import torch
from torch import Tensor

from lightly_train._commands import export_onnx
from lightly_train._commands.export_onnx import ExportONNXConfig

try:
    import onnx
except ImportError:
    pytest.skip("onnx is not installed", allow_module_level=True)

from .. import helpers


@pytest.fixture
def dummy_input() -> Tensor:
    """Fixture providing dummy input tensor."""

    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_checkpoint_path(tmp_path: Path) -> Path:
    """Fixture providing dummy backbone weights file."""

    weights_path = tmp_path / "dummy_weights.ckpt"
    # Create a dummy weights file
    dummy_weights = {
        "state_dict": {
            "backbone.weight": torch.randn(10, 10),
            "backbone.bias": torch.randn(10),
        }
    }
    torch.save(dummy_weights, weights_path)
    return weights_path


@pytest.fixture
def onnx_model_path(tmp_path: Path, dummy_checkpoint_path: Path) -> Path:
    """Fixture that creates an ONNX model file for testing."""

    onnx_path: Path = tmp_path / "model.onnx"
    export_onnx.export_onnx(
        out=onnx_path,
        checkpoint=dummy_checkpoint_path,
    )
    return onnx_path


def test_export_parameters() -> None:
    """Test that export function and configs have the same parameters and default values."""

    helpers.assert_same_params(a=ExportONNXConfig, b=export_onnx.export_onnx)


def test_export_succeeds(tmp_path: Path, dummy_checkpoint_path: Path) -> None:
    """Test that ONNX export succeeds and creates a valid file."""

    onnx_path: Path = tmp_path / "model.onnx"

    # Test export function
    export_onnx.export_onnx(
        out=onnx_path,
        checkpoint=dummy_checkpoint_path,
    )

    # Verify file exists and has content
    assert onnx_path.exists(), "ONNX file should be created"
    assert onnx_path.stat().st_size > 0, "ONNX file should not be empty"


def test_export_with_nonexistent_weights(tmp_path: Path) -> None:
    """Test that export fails gracefully with nonexistent weights file."""

    onnx_path: Path = tmp_path / "model.onnx"
    nonexistent_weights = tmp_path / "nonexistent_weights.pth"

    with pytest.raises((FileNotFoundError, RuntimeError)):
        export_onnx.export_onnx(
            out=onnx_path,
            checkpoint=nonexistent_weights,
        )


def test_export_onnx(onnx_model_path: Path) -> None:
    """Test that the exported ONNX model passes ONNX validation."""

    onnx_model: onnx.ModelProto = onnx.load(str(onnx_model_path))

    try:
        onnx.checker.check_model(onnx_model, full_check=True)
    except Exception as e:
        pytest.fail(f"ONNX model validation failed: {e}")


def test_onnxruntime_inference(dummy_input: Tensor, onnx_model_path: Path) -> None:
    """Test that ONNXRuntime can run inference on the exported model."""
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")
    try:
        ort_session = ort.InferenceSession(
            str(onnx_model_path), providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        pytest.fail(f"Failed to create ONNX Runtime session: {e}")

    try:
        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_session.run(["mask", "logits"], ort_inputs)
    except Exception as e:
        pytest.fail(f"ONNX Runtime inference failed: {e}")
