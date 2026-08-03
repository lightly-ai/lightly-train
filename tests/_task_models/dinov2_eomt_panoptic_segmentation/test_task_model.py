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

from lightly_train._task_models.dinov2_eomt_panoptic_segmentation.task_model import (
    DINOv2EoMTPanopticSegmentation,
)
from lightly_train._task_models.dinov2_eomt_panoptic_segmentation.train_model import (
    DINOv2EoMTPanopticSegmentationTrainArgs,
)


@pytest.fixture()
def model() -> DINOv2EoMTPanopticSegmentation:
    return DINOv2EoMTPanopticSegmentation(
        model_name="dinov2/_vittest14-eomt",
        thing_classes={0: "car", 1: "person"},
        stuff_classes={2: "sky", 3: "road"},
        image_size=(14, 14),
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        num_queries=2,
        num_joint_blocks=1,
        load_weights=False,
    )


def test_list_model_names__uses_registry() -> None:
    names = DINOv2EoMTPanopticSegmentation.list_model_names()

    assert "dinov2/_vittest14-eomt" in names
    assert "dinov2/vits14-eomt" in names
    assert "dinov2/vits14-notpretrained-eomt" in names


def test_is_supported_model__uses_registry() -> None:
    assert DINOv2EoMTPanopticSegmentation.is_supported_model("dinov2/vits14-eomt")
    assert DINOv2EoMTPanopticSegmentation.is_supported_model(
        "dinov2/vits14-notpretrained-eomt"
    )
    assert not DINOv2EoMTPanopticSegmentation.is_supported_model("dinov3/vits16-eomt")


@pytest.mark.parametrize(
    ("model_name", "expected_num_joint_blocks"),
    [
        ("dinov2/vits14-eomt", 3),
        ("dinov2/vitl14-eomt", 4),
        ("dinov2/vitg14-eomt", 5),
        ("dinov2/vitg14-notpretrained-eomt", 5),
    ],
)
def test_train_args_resolve_auto__uses_registry_num_joint_blocks(
    model_name: str, expected_num_joint_blocks: int
) -> None:
    args = DINOv2EoMTPanopticSegmentationTrainArgs()

    args.resolve_auto(
        total_steps=90_000,
        gradient_accumulation_steps=1,
        train_num_batches=10,
        model_name=model_name,
        model_init_args={},
        data_args=None,  # type: ignore[arg-type]
    )

    assert args.num_joint_blocks == expected_num_joint_blocks


def test_train_args_resolve_auto__model_init_args_override_registry() -> None:
    args = DINOv2EoMTPanopticSegmentationTrainArgs()

    args.resolve_auto(
        total_steps=90_000,
        gradient_accumulation_steps=1,
        train_num_batches=10,
        model_name="dinov2/vitg14-eomt",
        model_init_args={"num_joint_blocks": 2},
        data_args=None,  # type: ignore[arg-type]
    )

    assert args.num_joint_blocks == 2


def test_predict_batch__composes_stages_in_order(
    model: DINOv2EoMTPanopticSegmentation, mocker: MockerFixture
) -> None:
    preprocess_image_spy = mocker.spy(model, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model, "preprocess_batch")
    forward_backend_spy = mocker.spy(model, "forward_backend")
    postprocess_spy = mocker.spy(model, "postprocess")

    images = [torch.rand(3, 21, 28), torch.rand(3, 35, 21)]
    result = model.predict_batch(images=images)

    # Each input image goes through preprocess_image once.
    assert preprocess_image_spy.call_count == 2

    # The stacked batch is preprocessed in a single call with shape (B, C, H, W).
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert batch_in.shape == (2, 3, 14, 14)

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
def test_export_onnx(model: DINOv2EoMTPanopticSegmentation, tmp_path: Path) -> None:
    import onnx

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    output_names = [o.name for o in onnx_model.graph.output]
    assert "masks" in output_names
    assert "segment_ids" in output_names
    assert "scores" in output_names
