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


@pytest.mark.xfail(strict=True, reason="dinov3 ONNX export shape mismatch")
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx(model: DINOv3EoMTPanopticSegmentation, tmp_path: Path) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)


@pytest.mark.xfail(strict=True, reason="dinov3 ONNX export shape mismatch")
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__custom_height_width(
    model: DINOv3EoMTPanopticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, height=32, width=48, simplify=False, verify=True)
