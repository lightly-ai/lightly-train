#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import numpy as np


class MixUp:
    def __init__(
        self,
        *,
        prob: float,
        beta_range: tuple[float, float] = (0.45, 0.55),
    ) -> None:
        self.prob = prob
        self.beta_range = beta_range

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(batch) < 2 or np.random.random() >= self.prob:
            return batch

        images = np.stack([item["image"] for item in batch], axis=0)
        beta = round(np.random.uniform(*self.beta_range), 6)

        mixed_images = beta * images + (1.0 - beta) * np.roll(images, shift=1, axis=0)

        shifted_batch = list(batch[-1:]) + list(batch[:-1])
        mixed_batch: list[dict[str, Any]] = []
        for mixed_image, item, shifted_item in zip(mixed_images, batch, shifted_batch):
            mixed_batch.append(
                {
                    "image": mixed_image,
                    "bboxes": np.concatenate(
                        (item["bboxes"], shifted_item["bboxes"]), axis=0
                    ),
                    "class_labels": np.concatenate(
                        (item["class_labels"], shifted_item["class_labels"]), axis=0
                    ),
                }
            )
        return mixed_batch
