#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache

import lightly_train

from .. import helpers


@pytest.fixture(scope="module")
def dinov2_vits14_eomt_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tmp")
    directory = tmp
    out = directory / "out"
    train_images = directory / "train_images"
    train_masks = directory / "train_masks"
    val_images = directory / "val_images"
    val_masks = directory / "val_masks"
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
        num_workers=0,
        steps=1,
    )

    checkpoint_path = out / "checkpoints/last.ckpt"
    assert checkpoint_path.exists()
    return checkpoint_path


onnx_export_testset = [
    (1, 42, 154),
    (1, 42, 154),
    (2, 14, 14),
    (3, 140, 280),
    (4, 266, 28),
]


@pytest.mark.parametrize("batch_size,height,width", onnx_export_testset)
@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Requires Python 3.9 or higher for image preprocessing.",
)
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_onnx_export(
    batch_size: int,
    height: int,
    width: int,
    dinov2_vits14_eomt_checkpoint: Path,
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
    # We use  torch.testing.assert_close to check if the model outputs the same as when we run the exported
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
