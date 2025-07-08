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

from lightly_train._commands import train_task

from .. import helpers


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
        model="dinov2_vit/_vit_test14",
        task="semantic_segmentation",
        devices=1,
        batch_size=2,
        num_workers=2,
        steps=2,
    )
    assert out.exists()
    assert out.is_dir()
    assert (out / "train.log").exists()
