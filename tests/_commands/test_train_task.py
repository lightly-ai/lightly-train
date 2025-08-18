#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import pytest
from lightning_utilities.core.imports import RequirementCache

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)
if not RequirementCache("transformers"):
    pytest.skip("Transformers not installed", allow_module_level=True)

import os
import sys

import torch

import lightly_train

from .. import helpers

is_self_hosted_docker_runner = "GH_RUNNER_NAME" in os.environ


@pytest.mark.skipif(
    sys.platform.startswith("win") or is_self_hosted_docker_runner,
    reason=(
        "Fails on Windows since switching to Jaccard index "
        "OR on self-hosted CI with GPU (insufficient shared memory causes worker bus error)"
    ),
)
def test_train_semantic_segmentation(tmp_path: Path) -> None:
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    helpers.create_images(train_images)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_masks(val_masks)

    lightly_train.train_semantic_segmentation(
        out=out,
        data={
            "train": {
                "images": train_images,
                "masks": train_masks,
            },
            "val": {
                "images": val_images,
                "masks": val_masks,
            },
            "classes": {
                0: "background",
                1: "car",
            },
        },
        model="dinov2/_vittest14-eomt",
        model_args={
            "num_joint_blocks": 1,  # Reduce joint blocks for _vittest14
        },
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()

    model = lightly_train.load_model_from_checkpoint(
        checkpoint=out / "checkpoints" / "last.ckpt"
    )
    # Check forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    prediction = model.predict(dummy_input[0])
    assert prediction.shape == (224, 224)
    assert prediction.min() >= 0
    assert prediction.max() <= 1

    # Check ONNX export
    if RequirementCache("onnx"):
        import onnx

        onnx_out = out / "model.onnx"
        lightly_train.export_onnx(
            out=onnx_out,
            checkpoint=out / "checkpoints" / "last.ckpt",
        )
        onnx_model = onnx.load(str(onnx_out))
        onnx.checker.check_model(onnx_model, full_check=True)

        # Check ONNX inference
        if RequirementCache("onnxruntime"):
            import onnxruntime as ort

            ort_session = ort.InferenceSession(
                str(onnx_out), providers=["CPUExecutionProvider"]
            )
            ort_inputs = {"input": dummy_input.cpu().numpy()}
            onnx_masks, onnx_logits = ort_session.run(
                ["masks", "logits"], ort_inputs
            )
            assert onnx_masks.shape == (1, 224, 224)
            assert onnx_logits.shape == (1, 2, 224, 224)


@pytest.fixture(scope="module")
def dinov2_vits14_eomt_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tmp")
    out = tmp / "out"
    train_images = tmp / "train_images"
    train_masks = tmp / "train_masks"
    val_images = tmp / "val_images"
    val_masks = tmp / "val_masks"
    helpers.create_images(train_images)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_masks(val_masks)

    lightly_train.train_semantic_segmentation(
        out=out,
        data={
            "train": {
                "images": train_images,
                "masks": train_masks,
            },
            "val": {
                "images": val_images,
                "masks": val_masks,
            },
            "classes": {
                0: "background",
                1: "car",
            },
        },
        model="dinov2/vits14-eomt",
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=1,
    )

    checkpoint_path = out / "checkpoints/last.ckpt"
    assert checkpoint_path.exists()
    return checkpoint_path


test_dinov2_vits14_eomt_onnx_export_testset = [
    (1, 42, 154),
    (2, 14, 14),
    (3, 140, 280),
    (4, 266, 28),
]


@pytest.mark.parametrize(
    "batch_size,height,width", test_dinov2_vits14_eomt_onnx_export_testset
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_dinov2_vits14_eomt_onnx_export(
    batch_size: int,
    height: int,
    width: int,
    dinov2_vits14_eomt_checkpoint,
    tmp_path: Path,
) -> None:
    import onnx
    import onnxruntime as ort

    # arrange
    model = lightly_train.load_model_from_checkpoint(
        dinov2_vits14_eomt_checkpoint, device="cpu"
    )
    onnx_path = tmp_path / "model.onnx"
    validation_input = torch.randn(batch_size, 3, height, width).cpu()
    expected_outputs = model(validation_input)
    # We use  orch.testing.assert_close to check if the model outputs the same as when we run the exported
    # onnx file with onnxruntime. Unfortunately the default tolerances are too strict so we specify our own.
    rtol = 1e-3
    atol = 1e-5

    # act
    lightly_train.export_onnx(
        out=onnx_path,
        checkpoint=dinov2_vits14_eomt_checkpoint,
        height=height,
        width=width,
        batch_size=batch_size,
        overwrite=True,
    )

    # assert
    assert onnx_path.exists()
    onnx.checker.check_model(onnx_path, full_check=True)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_in = {"input": validation_input.numpy()}
    ort_outputs = session.run(["masks", "logits"], ort_in)
    ort_outputs = [torch.from_numpy(y).cpu() for y in ort_outputs]

    assert len(ort_outputs) == len(expected_outputs)
    for ort_y, expected_y in zip(ort_outputs, expected_outputs):
        torch.testing.assert_close(ort_y, expected_y, rtol=rtol, atol=atol)
