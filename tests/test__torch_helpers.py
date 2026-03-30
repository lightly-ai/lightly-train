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
from torch import nn

from lightly_train import _torch_helpers


class DummyClass:
    pass


def test__torch_weights_only_false(tmp_path: Path) -> None:
    fabric = Fabric(accelerator="cpu", devices=1)
    ckpt = {"dummy": DummyClass()}
    ckpt_path = tmp_path / "model.ckpt"
    fabric.save(ckpt_path, ckpt)  # type: ignore
    assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is None
    with _torch_helpers._torch_weights_only_false():
        assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") == "1"
        torch.load(ckpt_path)
        fabric.load(ckpt_path)
    assert os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") is None


def test_update_momentum() -> None:
    model = nn.Linear(2, 2, bias=False)
    model_ema = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        model_ema.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    ema_weight_ptr = model_ema.weight.data_ptr()
    _torch_helpers.update_momentum(model=model, model_ema=model_ema, m=0.25)

    expected = torch.tensor([[2.5, 3.5], [4.5, 5.5]])
    assert model_ema.weight.data_ptr() == ema_weight_ptr
    assert torch.equal(model_ema.weight, expected)


def test_update_ema_tensors() -> None:
    ema_tensors = [
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float64),
    ]
    tensors = [
        torch.tensor([5.0, 6.0], dtype=torch.float32),
        torch.tensor([7.0, 8.0], dtype=torch.float64),
    ]
    ema_tensor_ptrs = [tensor.data_ptr() for tensor in ema_tensors]

    _torch_helpers.update_ema_tensors(
        tensors=tensors,
        tensors_ema=ema_tensors,
        m=0.5,
    )

    assert [tensor.data_ptr() for tensor in ema_tensors] == ema_tensor_ptrs
    assert torch.equal(ema_tensors[0], torch.tensor([3.0, 4.0], dtype=torch.float32))
    assert torch.equal(ema_tensors[1], torch.tensor([5.0, 6.0], dtype=torch.float64))
