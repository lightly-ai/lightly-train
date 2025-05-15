#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random

import torch


def create_collated_masks(
    mask_ratio_tuple, mask_probability, n_crops, n_tokens, mask_generator=None
):
    n_masked_crops = int(n_crops * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_masked_crops + 1)

    upperbound = 0
    masks_list = []
    for i in range(0, n_masked_crops):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(n_tokens * random.uniform(prob_min, prob_max)))
            )
        )
        upperbound += int(n_tokens * prob_max)
    for i in range(n_masked_crops, n_crops):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)  # [n_crops, n_tokens]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )

    return {
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
    }
