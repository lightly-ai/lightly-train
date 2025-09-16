#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
from albumentations import BaseCompose, BasicTransform, SomeOf
from lightning_utilities.core.imports import RequirementCache
from numpy.typing import NDArray

ALBUMENTATIONS_GEQ_1_4_21 = RequirementCache("albumentations>=1.4.21")

if not ALBUMENTATIONS_GEQ_1_4_21:
    from albumentations import random_utils


class RandomOrder(SomeOf):  # type: ignore[misc]
    def __init__(
        self,
        transforms: list[BasicTransform | BaseCompose],
        n: int | None = None,
        replace: bool = False,
        p: float = 1.0,
    ):
        """Apply a random number of transformations from a list in random order.

        Args:
            transforms: List of transformations to choose from.
            n: How many transformations to sample. If None, len(transforms) is used.
            replace: Whether to sample transformations with replacement.
            p: Probability of applying the entire pipeline, not each individual transform.
        """
        if n is None:
            n = len(transforms)
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)

    def _get_idx(self) -> NDArray[np.int64]:
        if ALBUMENTATIONS_GEQ_1_4_21:
            return self.random_generator.choice(  # type: ignore[no-any-return]
                len(self.transforms),
                size=self.n,
                replace=self.replace,
            )
        else:
            return random_utils.choice(  # type: ignore[no-any-return]
                len(self.transforms),
                size=self.n,
                replace=self.replace,
                p=self.transforms_ps,
            )
