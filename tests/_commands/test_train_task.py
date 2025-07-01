#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from lightly_train._commands import train_task


def test_train_task(tmp_path: Path) -> None:
    train_task.train_task(
        out=tmp_path / "out",
        data={
            "train": {
                "images": tmp_path / "train_images",
                "masks": tmp_path / "train_masks",
            },
            "val": {
                "images": tmp_path / "val_images",
                "masks": tmp_path / "val_masks",
            },
            "classes": {
                0: "person",
                1: "car",
            },
        },
        model="dinov2_vit/vitb14",
        task="semantic_segmentation",
    )
