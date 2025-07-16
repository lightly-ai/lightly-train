#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from lightly_train._commands import export_task
from lightly_train._commands.export_task import ExportTaskConfig

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
    dummy_weights = {
        "state_dict": {
            "backbone.weight": torch.randn(10, 10),
            "backbone.bias": torch.randn(10),
        }
    }
    torch.save(dummy_weights, weights_path)
    return weights_path


class TestExportTask:
    def test_export_parameters(self) -> None:
        """Test that export function and configs have the same parameters and default values."""
        helpers.assert_same_params(a=ExportTaskConfig, b=export_task.export_task)

    def test_export__onnx(self, tmp_path: Path, dummy_checkpoint_path: Path) -> None:
        """Test that ONNX export succeeds and creates a valid file."""
        onnx_path = tmp_path / "model.onnx"

        export_task.export_task(
            out=onnx_path,
            checkpoint=dummy_checkpoint_path,
            format="onnx",
        )

        assert onnx_path.exists(), "ONNX file should be created"
        assert onnx_path.stat().st_size > 0, "ONNX file should not be empty"

    def test_export__invalid_weights(self, tmp_path: Path) -> None:
        """Test that export fails gracefully with invalid weights file."""
        onnx_path = tmp_path / "model.onnx"
        invalid_weights = tmp_path / "invalid_weights.pth"

        with pytest.raises((FileNotFoundError, RuntimeError)):
            export_task.export_task(
                out=onnx_path,
                checkpoint=invalid_weights,
                format="onnx",
            )

    def test_export__onnx_validation(
        self, tmp_path: Path, dummy_checkpoint_path: Path
    ) -> None:
        """Test that the exported ONNX model passes validation."""
        onnx_path = tmp_path / "model.onnx"

        export_task.export_task(
            out=onnx_path,
            checkpoint=dummy_checkpoint_path,
            format="onnx",
        )

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model, full_check=True)

    def test_export__inference_with_onnxruntime(
        self, tmp_path: Path, dummy_checkpoint_path: Path, dummy_input: Tensor
    ) -> None:
        """Test that ONNXRuntime can run inference on the exported model."""
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")

        onnx_path = tmp_path / "model.onnx"
        export_task.export_task(
            out=onnx_path,
            checkpoint=dummy_checkpoint_path,
            format="onnx",
        )

        ort_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_session.run(["mask", "logits"], ort_inputs)
