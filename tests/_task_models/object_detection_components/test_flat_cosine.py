from __future__ import annotations

import pytest
import torch
from torch import nn

from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)


def _make_scheduler(
    *,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    warmup_start_factor: float = 0.01,
) -> tuple[torch.optim.Optimizer, FlatCosineLRScheduler]:
    param = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = FlatCosineLRScheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
    )
    return optimizer, scheduler


def _advance(scheduler: FlatCosineLRScheduler, steps: int) -> None:
    for _ in range(steps):
        scheduler.step()


def test_flat_cosine_scheduler_phases() -> None:
    optimizer, scheduler = _make_scheduler()

    assert scheduler.has_cosine_phase
    assert scheduler.get_last_lr()[0] == pytest.approx(0.010099)

    _advance(scheduler, 99)
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0)

    _advance(scheduler, 1)
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0)

    _advance(scheduler, 454)
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0)

    _advance(scheduler, 1)
    cosine_lr = scheduler.get_last_lr()[0]
    assert 0.001 < cosine_lr < 1.0

    _advance(scheduler, 278)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001)

    _advance(scheduler, 100)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_flat_cosine_scheduler_state_dict_roundtrip() -> None:
    _, scheduler = _make_scheduler()
    _advance(scheduler, 123)
    state_dict = scheduler.state_dict()

    _, clone = _make_scheduler()
    clone.load_state_dict(state_dict)

    assert clone.last_step == scheduler.last_step
    assert clone.get_last_lr() == pytest.approx(scheduler.get_last_lr())

    scheduler.step()
    clone.step()
    assert clone.get_last_lr() == pytest.approx(scheduler.get_last_lr())
