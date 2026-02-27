#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from torchvision.transforms import v2


class RandomRotate90(v2.Transform):
    """
    Rotate an image by 90, 180, or 270 degrees with a given probability.

    Args:
        p: The probability with which the image will be rotated.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self._transform = v2.RandomApply(
            [
                v2.RandomChoice(
                    [
                        v2.RandomRotation(degrees=[90, 90]),
                        v2.RandomRotation(degrees=[180, 180]),
                        v2.RandomRotation(degrees=[270, 270]),
                        v2.Identity(),
                    ]
                )
            ],
            p=p,
        )

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt)
