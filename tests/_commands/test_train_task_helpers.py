#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import torch
from lightning_fabric import Fabric

from lightly_train._commands import train_task_helpers


class DummyClass:
    pass


def test__torch_weights_only_false(tmp_path: Path) -> None:
    fabric = Fabric(accelerator="cpu", devices=1)
    ckpt = {"dummy": DummyClass()}
    ckpt_path = tmp_path / "model.ckpt"
    fabric.save(ckpt_path, ckpt)  # type: ignore
    with train_task_helpers._torch_weights_only_false():
        torch.load(ckpt_path)
        fabric.load(ckpt_path)
