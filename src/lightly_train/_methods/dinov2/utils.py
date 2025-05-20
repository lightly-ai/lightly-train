#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
import random
from typing import Dict, List

import numpy as np
import torch


class MaskingGenerator:
    def __init__(
        self,
        input_size: int | tuple[int, int],
        max_num_patches: int,
        min_num_patches: int = 4,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self) -> str:
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self) -> tuple[int, int]:
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:  # type: ignore[type-arg]
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:  # type: ignore[type-arg]
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


def create_collated_masks(
    mask_ratio_min: float,
    mask_ratio_max: float,
    n_masked_crops: int,
    n_crops: int,
    mask_generator: MaskingGenerator,
) -> Dict[str, torch.Tensor]:
    n_patch_tokens = mask_generator.num_patches
    probs = np.linspace(mask_ratio_min, mask_ratio_max, n_masked_crops + 1)

    masks_list: List[torch.Tensor] = []
    for i in range(0, n_masked_crops):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(n_patch_tokens * random.uniform(prob_min, prob_max)))
            )
        )
    for i in range(n_masked_crops, n_crops):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)  # [G*B, H*W]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # [M,]
    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )  # [M,]

    return {
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
    }


def linear_warmup_schedule(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
) -> float:  # TODO: import from LightlySSL after new release
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if start_value < 0:
        raise ValueError(f"Start value {start_value} can't be negative.")
    if end_value <= 0:
        raise ValueError(f"End value {end_value} can't be non-positive.")
    if start_value > end_value:
        raise ValueError(
            f"Start value {start_value} must be less than or equal to end value {end_value}."
        )
    if step < warmup_steps:
        return start_value + step / warmup_steps * (end_value - start_value)
    else:
        return end_value
