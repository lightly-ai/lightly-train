#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
"""LT-DETR-style flat-cosine learning-rate scheduling."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# TODO: Replace these hardcoded reference values with public config when the
# LT-DETR / DEIMv2 recipe is published.
_REFERENCE_TOTAL_PHASE = 72
_REFERENCE_FLAT_PHASE = 29
_REFERENCE_NO_AUG_PHASE = 8
_REFERENCE_LR_GAMMA = 0.5


def flat_cosine_schedule(
    warmup_steps: int,
    cosine_start_step: int,
    cosine_end_step: int,
    current_step: int,
    init_lr: float,
    min_lr: float,
) -> float:
    """Compute the learning rate using a warm-up, flat, cosine, and tail schedule."""
    if warmup_steps > 0 and current_step <= warmup_steps:
        warmup_progress = current_step / float(warmup_steps)
        return init_lr * (warmup_progress**2)
    if current_step < cosine_start_step:
        return init_lr
    if current_step >= cosine_end_step:
        return min_lr
    cosine_decay = 0.5 * (
        1.0
        + math.cos(
            math.pi
            * (current_step - cosine_start_step)
            / (cosine_end_step - cosine_start_step)
        )
    )
    return min_lr + (init_lr - min_lr) * cosine_decay


class FlatCosineLRScheduler(LRScheduler):
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
        min_factor: float = _REFERENCE_LR_GAMMA,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError(f"min_factor must be between 0 and 1, got {min_factor}.")

        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, total_steps)
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
        if self.cosine_start_step >= self.cosine_end_step:
            raise ValueError(
                "flat-cosine scheduler requires a non-empty cosine phase, "
                f"but got total_steps={total_steps}, warmup_steps={warmup_steps}, "
                f"flat_steps={self.flat_steps}, no_aug_steps={self.no_aug_steps}, "
                f"cosine_start_step={self.cosine_start_step}, "
                f"cosine_end_step={self.cosine_end_step}."
            )

        self.min_lrs = [
            group["lr"] * self.min_factor for group in optimizer.param_groups
        ]

        super().__init__(optimizer, last_epoch)

    @property
    def last_step(self) -> int:
        return self.last_epoch

    def get_lr(self) -> list[float]:
        return [
            flat_cosine_schedule(
                warmup_steps=self.warmup_steps,
                cosine_start_step=self.cosine_start_step,
                cosine_end_step=self.cosine_end_step,
                current_step=self.last_epoch,
                init_lr=base_lr,
                min_lr=min_lr,
            )
            for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)
        ]
