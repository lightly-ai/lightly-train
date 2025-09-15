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
from numpy.typing import NDArray


class RandomOrder(SomeOf):  # type: ignore[misc]
    def __init__(
        self,
        transforms: list[BasicTransform | BaseCompose],
        n: int = 1,
        replace: bool = False,
        p: float = 1.0,
    ):
        super().__init__(transforms=transforms, n=n, replace=replace, p=p)

    def _get_idx(self) -> NDArray[np.int64]:
        return self.random_generator.choice(  # type: ignore[no-any-return]
            len(self.transforms), size=self.n, replace=self.replace
        )
