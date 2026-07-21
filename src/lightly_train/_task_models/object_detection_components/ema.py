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
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import math
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import Module

from lightly_train._torch_helpers import update_ema_tensors


class ModelEMA(Module):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmups: int = 2000,
    ):
        super().__init__()

        self.model = deepcopy(model).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.model.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        # Track the number of EMA updates as a buffer so it is saved in the model
        # state_dict and restored on resume. If it were a plain Python int it would
        # reset to 0 when resuming from a checkpoint, restarting the decay warmup ramp
        # (decay_fn(step=1) ~= 0) and overwriting the accumulated EMA weights with the
        # raw model weights on the first post-resume update. Since validation uses the
        # EMA weights, that produces a spurious drop in val metrics after resuming.
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.decay_fn = decay_fn  # decay exponential ramp (to help early epochs)

        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(
                decay=self.decay, warmup_steps=self.warmups, step=int(self.updates)
            )
            msd = model.state_dict()
            ema_tensors = []
            model_tensors = []
            for key, value in self.model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema_tensors.append(value)
                    model_tensors.append(msd[key].detach())
            update_ema_tensors(
                tensors=model_tensors,
                tensors_ema=ema_tensors,
                m=d,
            )

    def forward(
        self,
    ):
        raise RuntimeError("ema...")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"


def decay_fn(decay: float, warmup_steps: int, step: int) -> float:
    if warmup_steps <= 0:
        return decay
    else:
        return decay * (1 - math.exp(-step / warmup_steps))
