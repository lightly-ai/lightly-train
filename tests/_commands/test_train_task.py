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

from lightly_train._commands import export_task, train_task
from lightly_train._task_models import task_model_helpers

from .. import helpers

skip_on_ci_with_cuda = bool(os.environ.get("CI")) and torch.cuda.is_available()


@pytest.mark.skipif(
    sys.platform.startswith("win") or skip_on_ci_with_cuda,
    reason=(
        "Fails on Windows since switching to Jaccard index "
        "OR on self-hosted CI with GPU (insufficient shared memory causes worker bus error)"
    ),
)
def test_train_task(tmp_path: Path) -> None:
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    helpers.create_images(train_images)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_masks(val_masks)

    train_task.train_task(
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
        model="dinov2/_vittest14",
        task="semantic_segmentation",
        task_args={
            "num_joint_blocks": 1,  # Reduce joint blocks for _vittest14
        },
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()

    model = task_model_helpers.load_task_model_from_checkpoint(
        checkpoint=out / "checkpoints" / "last.ckpt"
    )
    # Check forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    model(dummy_input)

    # Check ONNX export
    if RequirementCache("onnx"):
        import onnx

        onnx_out = out / "model.onnx"
        export_task.export_task(
            out=onnx_out,
            checkpoint=out / "checkpoints" / "last.ckpt",
            format="onnx",
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
            ort_session.run(["mask", "logits"], ort_inputs)
