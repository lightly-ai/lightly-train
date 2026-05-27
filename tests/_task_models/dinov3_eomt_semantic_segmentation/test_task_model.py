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
from lightning_utilities.core.imports import RequirementCache
from pytest_mock import MockerFixture

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


def test_predict_batch__composes_stages_in_order(
    model: DINOv3EoMTSemanticSegmentation, mocker: MockerFixture
) -> None:
    preprocess_image_spy = mocker.spy(model, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model, "preprocess_batch")
    forward_backend_spy = mocker.spy(model, "forward_backend")
    postprocess_spy = mocker.spy(model, "postprocess")

    images = [torch.rand(3, 24, 32), torch.rand(3, 40, 24)]
    result = model.predict_batch(images=images)

    # Each input image goes through preprocess_image once.
    assert preprocess_image_spy.call_count == 2

    # preprocess_batch receives a sequence of per-image tensors with the same
    # short side (= min(image_size)) but different long sides.
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert len(batch_in) == 2
    assert all(t.shape[0] == 3 for t in batch_in)
    assert all(min(t.shape[-2:]) == min(model.image_size) for t in batch_in)

    # forward_backend receives the output of preprocess_batch.
    assert forward_backend_spy.call_count == 1
    (forward_in,) = forward_backend_spy.call_args.args
    assert forward_in is preprocess_batch_spy.spy_return

    # postprocess receives forward_backend's output and per-image metadata.
    assert postprocess_spy.call_count == 1
    raw_in, metadata = postprocess_spy.call_args.args
    assert raw_in is forward_backend_spy.spy_return
    assert len(metadata) == 2

    # predict_batch returns whatever postprocess produced.
    assert result is postprocess_spy.spy_return


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
    assert input_batch_dim.dim_param == "batch_size"

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


@pytest.mark.xfail(
    reason="ONNX export does not support non-square images yet (requires tiling support).",
    raises=ValueError,
    strict=True,
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__custom_height_width(
    model: DINOv3EoMTSemanticSegmentation, tmp_path: Path
) -> None:
    # TODO: Support ONNX export with non-square images (height != width).
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, height=32, width=48, simplify=False, verify=True)
