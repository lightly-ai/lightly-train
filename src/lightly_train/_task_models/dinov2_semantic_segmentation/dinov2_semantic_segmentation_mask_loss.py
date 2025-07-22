#
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
#

from __future__ import annotations

from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

# TODO(Guarin, 07/25): Move transformers classes into LightlyTrain. We should actually
# be able to use the original Mask2Former classes which don't depend on transformers:
# - https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
# - https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# If we replace transformers we'll still have to add SciPy as a dependency as parts
# of the loss depend on it.
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    Mask2FormerLoss,
)


class MaskClassificationLoss(Mask2FormerLoss):  # type: ignore[misc]
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable  # type: ignore[misc]
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict[str, torch.Tensor]],
        class_queries_logits: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)  # type: ignore[no-untyped-call]
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)  # type: ignore[arg-type]

        return {**loss_masks, **loss_classes}

    def loss_masks(self, masks_queries_logits, mask_labels, indices):  # type: ignore
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks_tensor = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks_tensor

        return loss_masks

    def loss_total(self, losses_all_layers: dict[str, torch.Tensor]) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)
        return loss_total  # type: ignore
