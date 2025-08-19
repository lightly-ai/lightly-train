#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class TwoStageWarmupPolySchedule(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        poly_power: float,
        last_epoch: int = -1,
    ):
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        lrs = []
        non_vit_warmup, vit_warmup = self.warmup_steps
        for i, base_lr in enumerate(self.base_lrs):
            if i >= self.num_backbone_params:
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            else:
                if step < non_vit_warmup:
                    lr = 0
                elif step < non_vit_warmup + vit_warmup:
                    lr = base_lr * ((step - non_vit_warmup) / vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup - vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup - vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            lrs.append(lr)
        return lrs
