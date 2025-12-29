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

try:
    import pydicom
except ImportError:
    pydicom = None  # type: ignore[assignment]


def test_train_object_detection(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_yolo_object_detection_dataset(data, split_first=True)

    # Check training
    lightly_train.train_object_detection(
        out=out,
        model="dinov3/vitt16-notpretrained-ltdetr",
        data={
            "path": data,
            "train": Path("train", "images"),
            "val": Path("val", "images"),
            "names": {
                0: "class_0",
                1: "class_1",
            },
        },
        steps=2,
        batch_size=2,
        num_workers=2,
        devices=1,
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()

    # Check that model can be loaded again
    model = lightly_train.load_model(model=out / "exported_models" / "exported_last.pt")

    # Check that only EMA weights are exported
    exported_state_dict = torch.load(
        out / "exported_models" / "exported_last.pt", map_location="cpu"
    )
    assert all(
        key.startswith("ema_model.")
        for key in exported_state_dict["train_model"].keys()
    )

    # Check forward pass
    dummy_input = torch.randn(3, 100, 200)
    results = model.predict(dummy_input)
    assert results["bboxes"].ndim == 2
    assert results["bboxes"].shape[1] == 4
    assert results["scores"].ndim == 1
    assert results["labels"].ndim == 1


@pytest.mark.skipif(
    is_self_hosted_docker_runner,
    reason=(
        "Fails on self-hosted CI with GPU (insufficient shared memory causes worker "
        "bus error"
    ),
)
def test_train_instance_segmentation(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_yolo_instance_segmentation_dataset(
        data, split_first=True, num_files=4
    )

    # Check training
    lightly_train.train_instance_segmentation(
        out=out,
        data={
            "path": data,
            "train": Path("train", "images"),
            "val": Path("val", "images"),
            "names": {
                0: "class_0",
                1: "class_1",
            },
        },
        model="dinov3/vitt16-notpretrained-eomt",
        model_args={"num_joint_blocks": 1},
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()

    # Check that the model can be loaded again
    model = lightly_train.load_model(model=out / "exported_models" / "exported_last.pt")

    # Check forward pass
    dummy_input = torch.randn(3, 100, 200)
    results = model.predict(dummy_input)

    assert results["labels"].ndim == 1
    assert results["masks"].ndim == 3
    assert results["masks"].shape[-2:] == dummy_input.shape[1:]
    assert results["scores"].ndim == 1


@pytest.mark.skipif(
    is_self_hosted_docker_runner,
    reason=(
        "Fails on self-hosted CI with GPU (insufficient shared memory causes worker "
        "bus error"
    ),
)
def test_train_panoptic_segmentation(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_coco_panoptic_segmentation_dataset(data, num_files=4)

    # Check training
    lightly_train.train_panoptic_segmentation(
        out=out,
        data={
            "train": {
                "images": data / "images" / "train",
                "masks": data / "annotations" / "train",
                "annotations": data / "annotations" / "train.json",
            },
            "val": {
                "images": data / "images" / "val",
                "masks": data / "annotations" / "val",
                "annotations": data / "annotations" / "val.json",
            },
        },
        model="dinov3/vitt16-notpretrained-eomt",
        model_args={"num_joint_blocks": 1},
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()

    # Check that the model can be loaded again
    model = lightly_train.load_model(model=out / "exported_models" / "exported_last.pt")

    # Check forward pass
    dummy_input = torch.randn(3, 100, 200)
    results = model.predict(dummy_input)
    assert results["masks"].shape == (100, 200, 2)
    assert results["segment_ids"].ndim == 1
    assert results["scores"].ndim == 1


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
    helpers.create_semantic_segmentation_masks(train_masks)
    helpers.create_images(val_images, num_channels=num_channels, mode=mode)
    helpers.create_semantic_segmentation_masks(val_masks)

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

    model = lightly_train.load_model(model=out / "exported_models" / "exported_last.pt")
    # Check forward pass
    dummy_input = torch.randn(1, num_channels, 224, 224)
    prediction = model.predict(dummy_input[0])
    assert prediction.shape == (224, 224)
    assert prediction.min() >= 0
    assert prediction.max() <= 1


@pytest.mark.skipif(pydicom is None, reason="pydicom not installed")
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Slow")
@pytest.mark.parametrize(
    ("data_format, num_channels, height, width"),
    [
        ("mr", 1, 64, 64),
        ("ct", 1, 128, 128),
        ("overlay", 1, 300, 484),
        ("rgb_color", 3, 240, 320),
        ("palette_color", 3, 350, 800),
        ("jpeg2k", 3, 480, 640),
    ],
)
def test_train_semantic_segmentation__dicom(
    tmp_path: Path,
    data_format: str,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    data_path: Path = pydicom_examples.get_path(data_format)

    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"

    train_images.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        image_filename = f"{index}_{data_path.name}"
        target = train_images / image_filename
        target.symlink_to(data_path)
    train_mask_filenames = [f"{index}_{data_path.stem}.png" for index in range(4)]
    helpers.create_semantic_segmentation_masks(
        train_masks,
        files=train_mask_filenames,
        height=height,
        width=width,
    )

    val_images.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        image_filename = f"{index}_{data_path.name}"
        target = val_images / image_filename
        target.symlink_to(data_path)
    val_mask_filenames = [f"{index}_{data_path.stem}.png" for index in range(2)]
    helpers.create_semantic_segmentation_masks(
        val_masks,
        files=val_mask_filenames,
        height=height,
        width=width,
    )

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
                1: "lesion",
            },
        },
        model="dinov2/_vittest14-eomt",
        model_args={"num_joint_blocks": 1},
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
def test_train_semantic_segmentation__export(
    tmp_path: Path, model_name: str, model_args: dict[str, Any], num_channels: int
) -> None:
    out = tmp_path / "out"
    train_images = tmp_path / "train_images"
    train_masks = tmp_path / "train_masks"
    val_images = tmp_path / "val_images"
    val_masks = tmp_path / "val_masks"
    mode = "RGB" if num_channels == 3 else "RGBA"
    helpers.create_images(train_images, num_channels=num_channels, mode=mode)
    helpers.create_semantic_segmentation_masks(train_masks)
    helpers.create_images(val_images, num_channels=num_channels, mode=mode)
    helpers.create_semantic_segmentation_masks(val_masks)

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
    ckpt_model_state_dict = lightly_train.load_model(
        out / "checkpoints" / "last.ckpt"
    ).state_dict()
    exported_model_state_dict = lightly_train.load_model(
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
    helpers.create_semantic_segmentation_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_semantic_segmentation_masks(val_masks)

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
        model="dinov2/_vittest14-eomt",
        model_args={"num_joint_blocks": 1},
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=0,
        steps=1,
    )
    last_ckpt_path = out / "exported_models" / "exported_last.pt"
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
            model=str(last_ckpt_path),
            model_args={"num_joint_blocks": 1},
            accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
            devices=1,
            batch_size=2,
            num_workers=0,
            steps=1,
            overwrite=True,
        )
    assert f"Loading model checkpoint from '{last_ckpt_path}'" in caplog.text

    # Part 3: check that the class head can be re-initialized when the number of classes differ.
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
                    2: "tree",
                },
            },
            model=str(last_ckpt_path),
            model_args={"num_joint_blocks": 1},
            accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
            devices=1,
            batch_size=2,
            num_workers=0,
            steps=1,
            overwrite=True,
        )
    assert "Checkpoint provides 2 classes but module expects 3." in caplog.text


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
    helpers.create_semantic_segmentation_masks(train_masks)
    helpers.create_images(val_images)
    helpers.create_semantic_segmentation_masks(val_masks)

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
        model="dinov2/_vittest14-eomt",
        model_args={"num_joint_blocks": 1},
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
            model="dinov2/_vittest14-eomt",
            model_args={"num_joint_blocks": 1},
            accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
            devices=1,
            batch_size=2,
            num_workers=0,
            steps=1,
            resume_interrupted=True,
        )

    assert f"Resuming training from model checkpoint '{last_ckpt_path}'" in caplog.text
    assert "Resuming training from step 1/1..." in caplog.text
