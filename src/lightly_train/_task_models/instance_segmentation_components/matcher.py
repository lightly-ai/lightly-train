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
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

# Modifications Copyright 2026 Lightly AG:
# - Added mask and dice matching costs on top of the RT-DETR box/class matcher.
# - Computed mask costs on randomly sampled points for memory efficiency.
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from torch import Tensor

from lightly_train._task_models.instance_segmentation_components.mask_ops import (
    dice_cost,
    point_sample,
    sigmoid_ce_cost,
)
from lightly_train._task_models.object_detection_components.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    sanitize_boxes_cxcywh_normalized,
)
from lightly_train._task_models.object_detection_components.matcher import (
    HungarianMatcher,
)

__all__ = ["MaskAwareHungarianMatcher"]


class MaskAwareHungarianMatcher(HungarianMatcher):  # type: ignore[misc]
    """Hungarian matcher for instance segmentation."""

    def __init__(
        self,
        weight_dict: dict[str, float],
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_points: int = 12544,
        mask_point_sample_ratio: int | None = None,
    ) -> None:
        """Creates the matcher.

        The classification cost always uses the focal-loss formulation, matching
        the upstream EdgeCrafter instance segmentation configuration.

        Args:
            weight_dict:
                Relative weights of the matching costs. Must contain the keys
                ``"cost_class"``, ``"cost_bbox"``, ``"cost_giou"``, ``"cost_mask"``,
                and ``"cost_dice"``. At least one weight must be non-zero.
            alpha:
                Focal-loss alpha balancing factor.
            gamma:
                Focal-loss gamma focusing factor.
            num_points:
                Number of points randomly sampled from each mask when computing the
                mask and dice costs. Ignored when ``mask_point_sample_ratio`` is set.
            mask_point_sample_ratio:
                If set, the number of sampled points is derived from the predicted
                mask resolution as
                ``max(height, height * width // mask_point_sample_ratio)``,
                reproducing the upstream EdgeCrafter behavior. When ``None`` the
                fixed ``num_points`` budget is used instead. Keep this equal to the
                criterion's ``mask_point_sample_ratio`` (upstream couples them).
        """
        # Initialize ``nn.Module`` directly instead of going through
        # ``HungarianMatcher.__init__``: the box-only superclass asserts that one
        # of ``cost_class``/``cost_bbox``/``cost_giou`` is non-zero, which would
        # reject valid mask/dice-only configurations before the extended
        # five-cost assertion below can run.
        torch.nn.Module.__init__(self)
        self.use_focal_loss: bool = True
        self.cost_class: float = weight_dict["cost_class"]
        self.cost_bbox: float = weight_dict["cost_bbox"]
        self.cost_giou: float = weight_dict["cost_giou"]
        self.cost_mask: float = weight_dict["cost_mask"]
        self.cost_dice: float = weight_dict["cost_dice"]
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.num_points: int = num_points
        self.mask_point_sample_ratio: int | None = mask_point_sample_ratio

        assert (
            self.cost_class != 0
            or self.cost_bbox != 0
            or self.cost_giou != 0
            or self.cost_mask != 0
            or self.cost_dice != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, list[tuple[Tensor, Tensor]]]:
        """Performs mask-aware bipartite matching.

        Mirrors the flattened-batch formulation of the upstream EdgeCrafter matcher:
        a single cost matrix is built over all queries and all targets from the
        class, bounding-box, GIoU, and (when masks are available) mask and dice
        costs, then split per batch element and solved with the linear sum
        assignment problem.

        Args:
            outputs:
                Dict of model outputs containing at least:
                    "pred_logits": Tensor of shape [batch_size, num_queries,
                        num_classes] with the classification logits.
                    "pred_boxes": Tensor of shape [batch_size, num_queries, 4] with
                        the predicted box coordinates in normalized cxcywh format.
                Optionally "pred_masks": Tensor of shape [batch_size, num_queries,
                    height, width] with the predicted mask logits. When absent (e.g.
                    for auxiliary encoder outputs) the mask and dice costs are
                    skipped.
            targets:
                List of length batch_size, where each target is a dict containing:
                    "labels": Tensor of shape [num_targets] with the class labels.
                    "boxes": Tensor of shape [num_targets, 4] with the target box
                        coordinates in normalized cxcywh format.
                    "masks": Tensor of shape [num_targets, height, width] with the
                        target masks. Only required when "pred_masks" is provided.

        Returns:
            A dict with key "indices" mapping to a list of length batch_size of
            (index_pred, index_tgt) tuples, where index_pred holds the indices of
            the matched predictions and index_tgt the indices of the corresponding
            targets. Batch elements with no targets yield a pair of empty tensors.
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten predictions to compute the cost matrices over the whole batch.
        cost_class = self._compute_class_cost(
            pred_logits=outputs["pred_logits"].flatten(0, 1),
            target_labels=torch.cat([target["labels"] for target in targets]),
        )
        # LightlyTrain sanitizes boxes before matching to stay consistent with the
        # box/GIoU losses in DFINECriterion (upstream skips this step).
        out_bbox = sanitize_boxes_cxcywh_normalized(outputs["pred_boxes"].flatten(0, 1))
        tgt_bbox = torch.cat([target["boxes"] for target in targets])
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(  # type: ignore[no-untyped-call]
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )
        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )

        # Auxiliary encoder outputs (``enc_aux_outputs``) carry no masks, so the
        # mask costs are only added when ``pred_masks`` is present.
        pred_masks = outputs.get("pred_masks")
        if pred_masks is not None and "masks" in targets[0]:
            cost_mask, cost_dice = self._compute_mask_costs(
                pred_masks=pred_masks.flatten(0, 1),
                target_masks=torch.cat([target["masks"] for target in targets]),
            )
            cost = cost + self.cost_mask * cost_mask + self.cost_dice * cost_dice

        # Handle potential NaNs in the cost matrix (e.g. under mixed precision),
        # matching the box-only HungarianMatcher.
        cost = torch.nan_to_num(cost.view(batch_size, num_queries, -1), nan=1.0).cpu()

        sizes = [len(target["boxes"]) for target in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(torch.split(cost, sizes, dim=-1))
        ]
        return {
            "indices": [
                (
                    torch.as_tensor(src_idx, dtype=torch.int64),
                    torch.as_tensor(tgt_idx, dtype=torch.int64),
                )
                for src_idx, tgt_idx in indices
            ]
        }

    def _compute_class_cost(
        self,
        pred_logits: Tensor,
        target_labels: Tensor,
    ) -> Tensor:
        """Computes the classification cost between predictions and targets.

        Args:
            pred_logits:
                Tensor of shape [num_queries, num_classes] with the classification
                logits for a single batch element.
            target_labels:
                Tensor of shape [num_targets] with the target class labels.

        Returns:
            Tensor of shape [num_queries, num_targets] with the focal-loss
            classification cost.
        """
        out_prob = pred_logits.sigmoid()
        out_prob = out_prob[:, target_labels]
        neg_cost_class = (
            (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = (
            self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
        )
        result = pos_cost_class - neg_cost_class
        if not isinstance(result, Tensor):
            raise TypeError("Expected focal class cost to be a tensor.")
        return result

    def _resolve_num_points(self, pred_masks: Tensor) -> int:
        """Resolves the number of points to sample for the mask costs.

        Derives the budget from the predicted-mask resolution when
        ``mask_point_sample_ratio`` is set (matching upstream EdgeCrafter) and
        falls back to the fixed ``num_points`` otherwise.

        Args:
            pred_masks:
                Tensor whose trailing two dimensions are the mask height and width.

        Returns:
            The number of points to sample.
        """
        if self.mask_point_sample_ratio is not None:
            height, width = pred_masks.shape[-2:]
            # Upstream floors the budget at one point per mask row, keeping the
            # matcher's sampling budget identical to the criterion's and avoiding
            # a zero-point sample (which would divide the mask/dice costs by zero).
            return max(height, height * width // self.mask_point_sample_ratio)
        return self.num_points

    def _compute_mask_costs(
        self,
        pred_masks: Tensor,
        target_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Computes the mask and dice costs between predictions and targets.

        Both costs are evaluated on the same set of randomly sampled point
        coordinates instead of the full masks to reduce memory usage.

        Args:
            pred_masks:
                Tensor of shape [batch_size * num_queries, height, width] with the
                flattened predicted mask logits.
            target_masks:
                Tensor of shape [num_targets, height, width] with the concatenated
                target masks across the batch.

        Returns:
            A tuple ``(cost_mask, cost_dice)`` of tensors of shape
            [batch_size * num_queries, num_targets], holding the sigmoid
            cross-entropy mask cost and the dice cost respectively.
        """
        point_coords = torch.rand(
            1,
            self._resolve_num_points(pred_masks),
            2,
            device=pred_masks.device,
            dtype=pred_masks.dtype,
        )
        pred_points = point_sample(
            input=pred_masks[:, None],
            point_coords=point_coords.expand(pred_masks.shape[0], -1, -1),
        ).squeeze(1)
        target_points = point_sample(
            input=target_masks[:, None].to(
                device=pred_masks.device,
                dtype=pred_masks.dtype,
            ),
            point_coords=point_coords.expand(target_masks.shape[0], -1, -1),
            mode="nearest",
        ).squeeze(1)

        return (
            sigmoid_ce_cost(inputs=pred_points, targets=target_points),
            dice_cost(inputs=pred_points, targets=target_points),
        )
