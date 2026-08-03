#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from Mask2Former (https://github.com/facebookresearch/Mask2Former)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

# Modifications Copyright 2026 Lightly AG:
# - Added point-sampled instance mask losses on top of the D-FINE criterion.
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lightly_train._task_models.instance_segmentation_components.mask_ops import (
    dice_loss,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sigmoid_ce_loss,
)
from lightly_train._task_models.object_detection_components.dfine_criterion import (
    DFINECriterion,
)

__all__ = ["EdgeCrafterInstanceSegmentationCriterion"]


class EdgeCrafterInstanceSegmentationCriterion(DFINECriterion):  # type: ignore[misc]
    """D-FINE criterion with point-sampled instance mask losses."""

    def __init__(
        self,
        *args: Any,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]
        # ``num_points`` fixes the point budget to the Mask2Former default.
        # EdgeCrafter derives the resolution-based budget from the matcher config,
        # keeping matching and mask-loss sampling coupled.
        mask_point_sample_ratio: int | None = getattr(
            self.matcher, "mask_point_sample_ratio", None
        )
        self.num_points = num_points
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if loss == "masks":
            if "pred_masks" not in outputs:
                return {}
            return self.loss_masks(
                outputs=outputs,
                targets=targets,
                indices=indices,
                num_boxes=num_boxes,
            )

        return super().get_loss(  # type: ignore[no-any-return, no-untyped-call]
            loss,
            outputs,
            targets,
            indices,
            num_boxes,
            **kwargs,
        )

    def loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        """Computes point-sampled sigmoid CE and dice losses for matched masks."""
        pred_masks = outputs["pred_masks"]
        zero_loss = pred_masks.sum() * 0.0
        if sum(len(src) for src, _ in indices) == 0:
            return {"loss_mask": zero_loss, "loss_dice": zero_loss}

        src_idx = self._get_src_permutation_idx(indices)  # type: ignore[no-untyped-call]
        matched_pred_masks = pred_masks[src_idx]
        matched_target_masks = torch.cat(
            [
                target["masks"][target_idx]
                for target, (_, target_idx) in zip(targets, indices)
            ],
            dim=0,
        ).to(device=matched_pred_masks.device, dtype=matched_pred_masks.dtype)

        if self.mask_point_sample_ratio is not None:
            height, width = matched_pred_masks.shape[-2:]
            # Upstream floors the budget at one point per mask row.
            num_points = max(height, height * width // self.mask_point_sample_ratio)
        else:
            num_points = self.num_points

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                logits=matched_pred_masks,
                num_points=num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            target_points = point_sample(
                input=matched_target_masks[:, None],
                point_coords=point_coords,
                mode="nearest",
            ).squeeze(1)
        pred_points = point_sample(
            input=matched_pred_masks[:, None],
            point_coords=point_coords,
        ).squeeze(1)

        return {
            "loss_mask": sigmoid_ce_loss(inputs=pred_points, targets=target_points)
            / num_boxes,
            "loss_dice": dice_loss(inputs=pred_points, targets=target_points)
            / num_boxes,
        }
