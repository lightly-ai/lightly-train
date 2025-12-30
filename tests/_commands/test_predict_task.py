#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import lightly_train

from .. import helpers

is_self_hosted_docker_runner = "GH_RUNNER_NAME" in os.environ


def create_dinov2_vits14_eomt_test_checkpoint(
    directory: Path, num_channels: int = 3
) -> Path:
    out = directory / "out"
    train_images = directory / "train_images"
    train_masks = directory / "train_masks"
    val_images = directory / "val_images"
    val_masks = directory / "val_masks"
    mode = "RGBA" if num_channels == 4 else "RGB"
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
        model="dinov2/vits14-eomt",
        transform_args={"num_channels": num_channels},
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        batch_size=2,
        num_workers=0,
        steps=1,
    )

    checkpoint_path = out / "exported_models" / "exported_last.pt"
    assert checkpoint_path.exists()
    return checkpoint_path


@pytest.fixture(scope="module")
def dinov2_vits14_eomt_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("tmp")
    return create_dinov2_vits14_eomt_test_checkpoint(directory=tmp)


@pytest.fixture(scope="module")
def dinov2_vits14_eomt_4_channels_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    tmp = tmp_path_factory.mktemp("tmp")
    return create_dinov2_vits14_eomt_test_checkpoint(directory=tmp, num_channels=4)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason=("Fails on Windows because of potential memory issues"),
)
@pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="Requires Python 3.9 or higher for image preprocessing.",
)
@pytest.mark.parametrize("num_channels", [3, 4])
def test_predict_semantic_segmentation(
    tmp_path: Path,
    num_channels: int,
    dinov2_vits14_eomt_checkpoint: Path,
    dinov2_vits14_eomt_4_channels_checkpoint: Path,
) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"

    mode = "RGB" if num_channels == 3 else "RGBA"
    num_images = 5
    helpers.create_images(data, num_channels=num_channels, mode=mode, files=num_images)

    checkpoint_path = {
        3: dinov2_vits14_eomt_checkpoint,
        4: dinov2_vits14_eomt_4_channels_checkpoint,
    }[num_channels]

    lightly_train.predict_semantic_segmentation(
        out=out,
        data=data,
        model=checkpoint_path,
        # The operator 'aten::upsample_bicubic2d.out' raises a NotImplementedError
        # on macOS with MPS backend.
        batch_size=1,
        num_workers=2,
        accelerator="auto" if not sys.platform.startswith("darwin") else "cpu",
        devices=1,
        num_channels=num_channels,
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "predict.log").exists()

    for i in range(num_images):
        assert (out / f"{i}.png").exists()
