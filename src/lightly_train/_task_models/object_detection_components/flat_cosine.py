"""LT-DETR-style flat-cosine learning-rate scheduling."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer

_REFERENCE_TOTAL_PHASE = 72
_REFERENCE_FLAT_PHASE = 40
_REFERENCE_NO_AUG_PHASE = 12


class FlatCosineLRScheduler:
    """Warmup + flat + cosine + final tail schedule.

    The flat and no-augmentation phases follow the LT-DETR / DEIMv2 reference
    recipe scaled to ``total_steps``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        *,
        warmup_start_factor: float = 0.0,
        min_factor: float = 0.001,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
        if not 0.0 <= warmup_start_factor <= 1.0:
            raise ValueError(
                f"warmup_start_factor must be between 0 and 1, got {warmup_start_factor}."
            )
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError(f"min_factor must be between 0 and 1, got {min_factor}.")

        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, total_steps)
        self.warmup_start_factor = warmup_start_factor
        self.min_factor = min_factor

        self.flat_steps = min(
            total_steps,
            max(
                0,
                math.floor(
                    total_steps * _REFERENCE_FLAT_PHASE / _REFERENCE_TOTAL_PHASE
                ),
            ),
        )
        self.no_aug_steps = min(
            total_steps,
            max(
                0,
                math.floor(
                    total_steps * _REFERENCE_NO_AUG_PHASE / _REFERENCE_TOTAL_PHASE
                ),
            ),
        )
        self.cosine_start_step = max(self.warmup_steps, self.flat_steps)
        self.cosine_end_step = max(
            self.cosine_start_step, total_steps - self.no_aug_steps
        )
        self.has_cosine_phase = self.cosine_start_step < self.cosine_end_step

        self.last_step = -1
        self._last_lr = list(self.base_lrs)
        self.step()

    def state_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        for lr, param_group in zip(self._last_lr, self.optimizer.param_groups):
            param_group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def step(self) -> None:
        self.last_step += 1
        factor = self._lr_factor(self.last_step)
        self._last_lr = []
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = base_lr * factor
            param_group["lr"] = lr
            self._last_lr.append(lr)

    def _lr_factor(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            progress = (step + 1) / self.warmup_steps
            return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * (
                progress**2
            )

        if not self.has_cosine_phase:
            return self.min_factor

        if step < self.cosine_start_step:
            return 1.0
        if step >= self.cosine_end_step:
            return self.min_factor

        cosine_steps = self.cosine_end_step - self.cosine_start_step
        progress = (step - self.cosine_start_step + 1) / cosine_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine
