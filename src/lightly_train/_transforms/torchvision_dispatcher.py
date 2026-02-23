#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
from torchvision.transforms import v2

from lightly_train._transforms.scale_jitter import generate_discrete_sizes


class SeededRandomChoice(v2.Transform):
    def __init__(self, transforms: Sequence[v2.Transform], seed: int) -> None:
        super().__init__()
        self.transforms = list(transforms)
        self.generator = torch.Generator().manual_seed(seed)
        self._current_idx = self._generate_idx()

    def _generate_idx(self) -> int:
        return int(
            torch.randint(
                0, len(self.transforms), (1,), generator=self.generator
            ).item()
        )

    def step(self) -> None:
        self._current_idx = self._generate_idx()

    def forward(self, *inputs):
        return self.transforms[self._current_idx](*inputs)


class TorchVisionScaleJitter(v2.Transform):
    def __init__(
        self,
        *,
        sizes: Sequence[tuple[int, int]] | None = None,
        target_size: tuple[int, int] | None = None,
        scale_range: tuple[float, float] | None = None,
        num_scales: int | None = None,
        divisible_by: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        sizes_list = generate_discrete_sizes(
            sizes=sizes,
            target_size=target_size,
            scale_range=scale_range,
            num_scales=num_scales,
            divisible_by=divisible_by,
        )

        transforms = [v2.Resize(size=(h, w), antialias=True) for h, w in sizes_list]

        self._transform = SeededRandomChoice(transforms, seed=seed)

    def forward(self, *inputs):
        return self._transform(*inputs)

    @contextmanager
    def same_seed(self) -> Iterator[None]:
        self._transform.step()
        yield
