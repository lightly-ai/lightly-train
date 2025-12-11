#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
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
    assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is None
    with train_task_helpers._torch_weights_only_false():
        assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
        torch.load(ckpt_path)
        fabric.load(ckpt_path)
    assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is None
