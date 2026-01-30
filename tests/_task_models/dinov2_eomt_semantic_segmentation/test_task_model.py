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
from lightning_utilities.core.imports import RequirementCache

from lightly_train._task_models.dinov2_eomt_semantic_segmentation.task_model import (
    DINOv2EoMTSemanticSegmentation,
)


@pytest.fixture()
def model() -> DINOv2EoMTSemanticSegmentation:
    return DINOv2EoMTSemanticSegmentation(
        model_name="dinov2/_vittest14-eomt",
        classes={0: "background", 1: "car"},
        class_ignore_index=None,
        image_size=(14, 14),
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        num_queries=2,
        num_joint_blocks=1,
        load_weights=False,
    )


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__dynamic_batch_size(
    model: DINOv2EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, dynamic_batch_size=True, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_batch_dim.dim_param == "N"

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inputs = np.random.randn(3, 3, 14, 14).astype(np.float32)
    session.run(None, {"images": inputs})


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__static_batch_size(
    model: DINOv2EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(
        out=out, batch_size=3, dynamic_batch_size=False, simplify=False, verify=True
    )


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__custom_height_width(
    model: DINOv2EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, height=28, width=42, simplify=False, verify=True)
