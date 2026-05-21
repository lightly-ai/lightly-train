#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)


def _make_scheduler(
    *,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    flat_steps: int = 555,
    no_aug_steps: int = 166,
) -> tuple[Optimizer, FlatCosineLRScheduler]:
    param = nn.Parameter(torch.ones(()))
    optimizer = SGD([param], lr=1.0)
    scheduler = FlatCosineLRScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        flat_steps=flat_steps,
        no_aug_steps=no_aug_steps,
    )
    return optimizer, scheduler


def _advance(
    optimizer: Optimizer, scheduler: FlatCosineLRScheduler, steps: int
) -> None:
    for _ in range(steps):
        optimizer.step()
        scheduler.step()


def test_flat_cosine_scheduler_phases() -> None:
    optimizer, scheduler = _make_scheduler()

    assert scheduler.flat_steps == 555
    assert scheduler.no_aug_steps == 166
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)

    _advance(optimizer, scheduler, 100)
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0)

    _advance(optimizer, scheduler, 455)
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0)

    _advance(optimizer, scheduler, 1)
    cosine_lr = scheduler.get_last_lr()[0]
    assert 0.5 < cosine_lr < 1.0

    _advance(optimizer, scheduler, 277)
    assert 0.5 < scheduler.get_last_lr()[0] < cosine_lr

    _advance(optimizer, scheduler, 1)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.5)

    _advance(optimizer, scheduler, 166)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_flat_cosine_scheduler_rejects_collapsed_cosine_phase() -> None:
    with pytest.raises(ValueError, match="non-empty cosine phase"):
        _make_scheduler(total_steps=1000, warmup_steps=1000)


def test_flat_cosine_scheduler_state_dict_roundtrip() -> None:
    optimizer, scheduler = _make_scheduler()
    _advance(optimizer, scheduler, 123)
    state_dict = scheduler.state_dict()  # type: ignore[no-untyped-call]

    clone_optimizer, clone_scheduler = _make_scheduler()
    clone_scheduler.load_state_dict(state_dict)

    assert clone_scheduler.last_step == scheduler.last_step
    assert clone_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())

    optimizer.step()
    scheduler.step()
    clone_optimizer.step()
    clone_scheduler.step()
    assert clone_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
