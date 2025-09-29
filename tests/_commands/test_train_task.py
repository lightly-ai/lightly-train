#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lightning_utilities.core.imports import RequirementCache
from pytest import LogCaptureFixture

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)
if not RequirementCache("transformers"):
    pytest.skip("Transformers not installed", allow_module_level=True)

import logging
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
@pytest.mark.parametrize(
    "model_name, model_args",
    [
        # Reduce number of joint blocks _vittest14.
        ("dinov2/_vittest14-eomt", {"num_joint_blocks": 1}),
        ("dinov2/_vittest14-linear", {}),
    ],
)
@pytest.mark.parametrize("num_channels", [3, 4])
def test_train_semantic_segmentation(
    tmp_path: Path, model_name: str, model_args: dict[str, Any], num_channels: int
) -> None:
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    mode = "RGB" if num_channels == 3 else "RGBA"
    helpers.create_images(train_images, num_channels=num_channels, mode=mode)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images, num_channels=num_channels, mode=mode)
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
        model=model_name,
        model_args=model_args,
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
        transform_args={
            "num_channels": num_channels,
        },
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()

    model = lightly_train.load_model_from_checkpoint(
        checkpoint=out / "exported_models" / "exported_last.pt"
    )
    # Check forward pass
    dummy_input = torch.randn(1, num_channels, 224, 224)
    prediction = model.predict(dummy_input[0])
    assert prediction.shape == (224, 224)
    assert prediction.min() >= 0
    assert prediction.max() <= 1


@pytest.mark.skipif(
    sys.platform.startswith("win") or is_self_hosted_docker_runner,
    reason=(
        "Fails on Windows since switching to Jaccard index "
        "OR on self-hosted CI with GPU (insufficient shared memory causes worker bus error)"
    ),
)
@pytest.mark.parametrize(
    "model_name, model_args",
    [
        # Reduce number of joint blocks _vittest14.
        ("dinov2/_vittest14-eomt", {"num_joint_blocks": 1}),
        ("dinov2/_vittest14-linear", {}),
    ],
)
@pytest.mark.parametrize("num_channels", [3, 4])
def test_train_semantic_segmentation_export(
    tmp_path: Path, model_name: str, model_args: dict[str, Any], num_channels: int
) -> None:
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    mode = "RGB" if num_channels == 3 else "RGBA"
    helpers.create_images(train_images, num_channels=num_channels, mode=mode)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images, num_channels=num_channels, mode=mode)
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
        model=model_name,
        model_args=model_args,
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
        transform_args={
            "num_channels": num_channels,
        },
    )

    # Check that last.ckpt and exported_model.pt contain same information.
    ckpt_model_state_dict = lightly_train.load_model_from_checkpoint(
        out / "checkpoints" / "last.ckpt"
    ).state_dict()
    exported_model_state_dict = lightly_train.load_model_from_checkpoint(
        out / "exported_models" / "exported_last.pt"
    ).state_dict()
    assert ckpt_model_state_dict.keys() == exported_model_state_dict.keys()
    for key in ckpt_model_state_dict.keys():
        assert torch.equal(ckpt_model_state_dict[key], exported_model_state_dict[key])


@pytest.mark.skipif(
    sys.platform.startswith("win") or is_self_hosted_docker_runner,
    reason=(
        "Fails on Windows since switching to Jaccard index "
        "OR on self-hosted CI with GPU (insufficient shared memory causes worker bus error)"
    ),
)
def test_train_semantic_segmentation__checkpoint(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Assert that load_checkpoint_from_file is called when a checkpoint is provided."""
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    helpers.create_images(train_images)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_masks(val_masks)

    # Part 1: Generate a checkpoint.
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
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    assert last_ckpt_path.exists()

    # Part 2: Load the checkpoint via the checkpoint parameter and assert log.
    with caplog.at_level(logging.INFO):
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
            accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
            devices=1,
            batch_size=2,
            num_workers=0,
            steps=1,
            overwrite=True,
            checkpoint=last_ckpt_path,
        )
    assert f"Loading checkpoint from '{last_ckpt_path}'" in caplog.text


@pytest.mark.skipif(
    sys.platform.startswith("win") or is_self_hosted_docker_runner,
    reason=(
        "Fails on Windows since switching to Jaccard index "
        "OR on self-hosted CI with GPU (insufficient shared memory causes worker bus error)"
    ),
)
def test_train_semantic_segmentation__resume_interrupted(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    """Assert that resume_interrupted loads the last checkpoint from the output dir."""
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    helpers.create_images(train_images)
    helpers.create_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_masks(val_masks)

    # Part 1: Generate a checkpoint that can be resumed.
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
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=0,
        steps=1,
    )
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    assert last_ckpt_path.exists()

    # Part 2: Resume from the generated checkpoint without providing ckpt explicitly.
    caplog.clear()
    with caplog.at_level(logging.INFO):
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
            accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
            devices=1,
            batch_size=2,
            num_workers=0,
            steps=1,
            resume_interrupted=True,
        )

    assert f"Loading checkpoint from '{last_ckpt_path}'" in caplog.text
    assert "Resuming training from step 1/1..." in caplog.text
