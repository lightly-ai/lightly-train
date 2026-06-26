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
from typing import cast

import torch
from lightning_fabric import Fabric
from torch import Tensor, nn

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


def test_total_gradient_norm__returns_total_norm() -> None:
    """total_gradient_norm returns the L2 norm of all parameter gradients.

    Mirrors torch.nn.utils.clip_grad_norm_ in its treatment of missing grads:
    parameters with `grad is None` are skipped, the rest contribute their sum
    of squares to the total.
    """
    # Grad on p1 contributes ||[1, 0]||^2 = 1.
    p1 = nn.Parameter(torch.tensor([3.0, 4.0]))
    p1.grad = torch.tensor([1.0, 0.0])
    # Grad on p2 contributes ||[0, 1]||^2 = 1.
    p2 = nn.Parameter(torch.tensor([0.0, 12.0]))
    p2.grad = torch.tensor([0.0, 1.0])
    # p3 has no grad and must be skipped.
    p3 = nn.Parameter(torch.tensor([5.0]))
    # Grad on p4 contributes ||[3]||^2 = 9.
    p4 = nn.Parameter(torch.tensor([5.0]))
    p4.grad = torch.tensor([3.0])

    grad_snapshots = [cast(Tensor, p.grad).detach().clone() for p in (p1, p2, p4)]

    norm = _torch_helpers.total_gradient_norm([p1, p2, p3, p4])

    # total ||grad|| = sqrt(1 + 1 + 9) = sqrt(11)
    assert torch.allclose(norm, torch.tensor(11.0).sqrt())
    # Function must not mutate any input gradient.
    for p, snapshot in zip((p1, p2, p4), grad_snapshots):
        assert torch.equal(cast(Tensor, p.grad), snapshot)


def test_total_gradient_norm__returns_zero_when_no_grads() -> None:
    """total_gradient_norm returns 0.0 when no parameter has a gradient."""
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    assert p.grad is None

    norm = _torch_helpers.total_gradient_norm([p])

    assert float(norm) == 0.0


def test_total_gradient_norm__non_cpu_grad_device() -> None:
    """Regression for mrpositron's review on PR #811.

    Previously the helper initialized the accumulator as `torch.zeros(1)` on CPU
    and added `p.grad.detach().pow(2).sum()` to it, which raises a device
    mismatch on GPU / MPS / meta. The fix lazily initializes the accumulator
    from the first parameter's grad so it lives on the same device as the
    gradients. We use the `meta` device so this regression test runs on
    CPU-only CI (no GPU required). Elementwise comparison is not possible on
    meta tensors (no `.item()`), so we only assert the device matches and the
    result is a scalar; correctness on CPU is covered by
    `test_total_gradient_norm__returns_total_norm`.
    """
    p1 = nn.Parameter(torch.empty(2, device="meta"))
    p1.grad = torch.zeros(2, device="meta")
    p2 = nn.Parameter(torch.empty(3, device="meta"))
    p2.grad = torch.ones(3, device="meta")
    p3 = nn.Parameter(torch.empty(4, device="meta"))  # no grad, skipped

    norm = _torch_helpers.total_gradient_norm([p1, p2, p3])

    assert norm.device.type == "meta"
    assert norm.shape == torch.Size([])
    assert norm.dtype.is_floating_point
