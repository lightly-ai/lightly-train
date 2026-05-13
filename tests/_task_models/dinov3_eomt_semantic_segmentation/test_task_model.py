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

from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import (
    DINOv3EoMTSemanticSegmentation,
)


@pytest.fixture()
def model() -> DINOv3EoMTSemanticSegmentation:
    return DINOv3EoMTSemanticSegmentation(
        model_name="dinov3/_vittest16-eomt",
        classes={0: "background", 1: "car"},
        class_ignore_index=None,
        image_size=(16, 16),
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
    model: DINOv3EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, dynamic_batch_size=True, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_batch_dim.dim_param == "N"

    import torch

    inputs = np.random.randn(3, 3, 16, 16).astype(np.float32)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": inputs})

    with torch.no_grad():
        torch_outputs = model(torch.from_numpy(inputs))

    for onnx_out, torch_out in zip(onnx_outputs, torch_outputs):
        onnx_tensor = torch.from_numpy(onnx_out)
        if torch_out.is_floating_point():
            close = torch.isclose(onnx_tensor, torch_out, atol=2e-2, rtol=1e-1)
            assert close.float().mean() > 0.95
        else:
            assert (onnx_tensor == torch_out).float().mean() > 0.95


@pytest.mark.xfail(
    strict=True, reason="dinov3 ONNX export shape mismatch with static batch size"
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__static_batch_size(
    model: DINOv3EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(
        out=out, batch_size=3, dynamic_batch_size=False, simplify=False, verify=True
    )


@pytest.mark.xfail(strict=True, reason="dinov3 ONNX export shape mismatch")
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__custom_height_width(
    model: DINOv3EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, height=32, width=48, simplify=False, verify=True)
