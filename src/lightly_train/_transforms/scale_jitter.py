#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from albumentations import Resize
from albumentations.core.transforms_interface import DualTransform

from lightly_train.types import NDArrayBBoxes, NDArrayImage, NDArrayMask


class ScaleJitter(DualTransform):  # type: ignore[misc]
    def __init__(
        self,
        *,
        sizes: Sequence[tuple[int, int]] | None,
        target_size: tuple[int, int] | None = None,
        scale_range: tuple[float, float] | None = None,
        num_scales: int | None = None,
        divisible_by: int | None = None,
        p: float = 1.0,
        step_seeding: bool = False,
        seed_offset: int = 0,
    ):
        super().__init__(p=1.0)
        if sizes is not None and any(
            [s is not None for s in [target_size, scale_range, num_scales]]
        ):
            raise ValueError(
                "If sizes is provided, target_size, scale_range, num_scales must be None."
            )
        if sizes is None and any(
            [s is None for s in [target_size, scale_range, num_scales]]
        ):
            raise ValueError(
                "If sizes is not provided, target_size, scale_range and num_scales must be provided."
            )
        self.sizes = sizes
        self.target_size = target_size
        self.scale_range = scale_range
        self.divisible_by = divisible_by
        self.p = p
        self.seed_offset = seed_offset
        self.step_seeding = step_seeding

        self._step = 0

        if not sizes:
            assert target_size is not None
            assert scale_range is not None
            assert num_scales is not None
            factors = np.linspace(
                start=scale_range[0],
                stop=scale_range[1],
                num=num_scales,
            )
            self.heights = (factors * target_size[0]).astype(np.int64)
            self.widths = (factors * target_size[1]).astype(np.int64)
        else:
            self.heights = np.array([s[0] for s in sizes], dtype=np.int64)
            self.widths = np.array([s[1] for s in sizes], dtype=np.int64)

        if divisible_by is not None:
            self.heights = (
                np.round(self.heights / divisible_by) * divisible_by
            ).astype(np.int64)
            self.widths = (np.round(self.widths / divisible_by) * divisible_by).astype(
                np.int64
            )

        self.transforms = [
            Resize(height=h, width=w) for h, w in zip(self.heights, self.widths)
        ]

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, step: int) -> None:
        self._step = step

    def get_params(self) -> dict[str, Any]:
        if self.step_seeding:
            rng = np.random.default_rng(self.step + self.seed_offset)
            idx = int(rng.integers(0, len(self.transforms)))
            return {"idx": idx}
        else:
            idx = int(np.random.randint(0, len(self.transforms)))
            return {"idx": idx}

    def apply(self, img: NDArrayImage, idx: int, **params: Any) -> NDArrayImage:
        return self.transforms[idx].apply(img=img, **params)  # type: ignore[no-any-return]

    def apply_to_bboxes(
        self, bboxes: NDArrayBBoxes, idx: int, **params: Any
    ) -> NDArrayBBoxes:
        return self.transforms[idx].apply_to_bboxes(bboxes, **params)  # type: ignore[no-any-return]

    def apply_to_mask(self, mask: NDArrayMask, idx: int, **params: Any) -> NDArrayMask:
        return self.transforms[idx].apply_to_mask(mask, **params)  # type: ignore[no-any-return]
