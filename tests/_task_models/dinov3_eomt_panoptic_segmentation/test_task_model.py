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

from lightly_train._export.onnx_helpers import (
    _TORCH_DYNAMO_AVAILABLE,
    _TORCH_DYNAMO_MIN_VERSION,
)
from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model import (
    DINOv3EoMTPanopticSegmentation,
)


@pytest.fixture()
def model() -> DINOv3EoMTPanopticSegmentation:
    return DINOv3EoMTPanopticSegmentation(
        model_name="dinov3/_vittest16-eomt",
        thing_classes={0: "car", 1: "person"},
        stuff_classes={2: "sky", 3: "road"},
        image_size=(16, 16),
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        num_queries=2,
        num_joint_blocks=1,
        load_weights=False,
    )


def test_predict_batch__composes_stages_in_order(
    model: DINOv3EoMTPanopticSegmentation, mocker: MockerFixture
) -> None:
    preprocess_image_spy = mocker.spy(model, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model, "preprocess_batch")
    forward_backend_spy = mocker.spy(model, "forward_backend")
    postprocess_spy = mocker.spy(model, "postprocess")

    images = [torch.rand(3, 24, 32), torch.rand(3, 40, 24)]
    result = model.predict_batch(images=images)

    # Each input image goes through preprocess_image once.
    assert preprocess_image_spy.call_count == 2

    # The stacked batch is preprocessed in a single call with shape (B, C, H, W).
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert batch_in.shape == (2, 3, 16, 16)

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


@pytest.mark.skipif(
    not _TORCH_DYNAMO_AVAILABLE, reason=f"torch >= {_TORCH_DYNAMO_MIN_VERSION} required"
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx(model: DINOv3EoMTPanopticSegmentation, tmp_path: Path) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)


@pytest.mark.long_running_test
@pytest.mark.skipif(
    not _TORCH_DYNAMO_AVAILABLE, reason=f"torch >= {_TORCH_DYNAMO_MIN_VERSION} required"
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__dynamic_output_shapes(
    tmp_path: Path,
) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from PIL import Image
    from torchvision.transforms.v2 import functional as F

    import lightly_train

    model = lightly_train.load_model("dinov3/vitt16-eomt-panoptic-coco", device="cpu")
    assert isinstance(model, DINOv3EoMTPanopticSegmentation)

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    # Verify that segment_ids and scores have a dynamic num_segments dimension
    # in the ONNX graph.
    onnx_model = onnx.load(out)
    outputs_by_name = {o.name: o for o in onnx_model.graph.output}
    for name in ("segment_ids", "scores"):
        dim = outputs_by_name[name].type.tensor_type.shape.dim[1]
        assert dim.dim_param != "", f"Expected dynamic dim 1 for output '{name}'"

    # Verify that different inputs produce different num_segments at runtime,
    # confirming the dimension is truly dynamic.
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    h, w = model.image_size
    mean = model.image_normalize["mean"]
    std = model.image_normalize["std"]

    def preprocess(img: Image.Image) -> np.ndarray:
        t = F.to_dtype(F.to_image(img), torch.float32, scale=True)
        t = F.resize(t, [h, w])
        t = F.normalize(t, mean=list(mean), std=list(std))
        return np.asarray(t.unsqueeze(0))

    # Real image with objects — should produce many segments.
    real_image_path = (
        Path(__file__).resolve().parents[2] / "test_images" / "Peppers.png"
    )
    scene_input = preprocess(Image.open(real_image_path).convert("RGB"))
    # Uniform gray image — should produce few segments.
    uniform_input = preprocess(Image.new("RGB", (h, w), (128, 128, 128)))

    scene_outputs = session.run(None, {"images": scene_input})
    uniform_outputs = session.run(None, {"images": uniform_input})

    scene_num_segments = scene_outputs[1].shape[1]
    uniform_num_segments = uniform_outputs[1].shape[1]
    assert scene_num_segments != uniform_num_segments, (
        f"Expected different num_segments for different inputs, "
        f"but both produced {scene_num_segments}"
    )


@pytest.mark.skipif(
    not _TORCH_DYNAMO_AVAILABLE, reason=f"torch >= {_TORCH_DYNAMO_MIN_VERSION} required"
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__custom_height_width(
    model: DINOv3EoMTPanopticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, height=32, width=48, simplify=False, verify=True)
