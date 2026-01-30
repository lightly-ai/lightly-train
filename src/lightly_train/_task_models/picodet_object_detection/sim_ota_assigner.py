#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torch import Tensor

from lightly_train._task_models.picodet_object_detection.losses import (
    box_iou,
    generalized_box_iou,
)


class TaskAlignedTop1Assigner:
    """Task-aligned top-1 assigner for one-to-one matching.

    This assigns each ground-truth to at most one prediction using a
    GT-centric top-1 selection based on a task-aligned metric with
    collision resolution.

    Args:
        alpha: Power for classification score in the metric.
        beta: Power for IoU in the metric.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 6.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    @torch.no_grad()
    def assign(
        self,
        *,
        pred_boxes_xyxy: Tensor,
        pred_cls_logits: Tensor,
        gt_boxes_xyxy: Tensor,
        gt_labels: Tensor,
        prior_centers: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Assign predictions to ground truth.

        Args:
            pred_boxes_xyxy: Predicted boxes (N, 4) in xyxy pixel coords.
            pred_cls_logits: Predicted class logits (N, C).
            gt_boxes_xyxy: Ground-truth boxes (M, 4) in xyxy pixel coords.
            gt_labels: Ground-truth labels (M,).
            prior_centers: Optional prior centers (N, 2) in pixel coords for
                spatial prior masking (center inside gt).

        Returns:
            Tuple of:
            - assigned_gt_index: (N,) index of matched GT or -1.
            - assigned_labels: (N,) matched labels or -1.
            - assigned_ious: (N,) IoU for matched predictions.
        """
        device = pred_boxes_xyxy.device
        num_preds = pred_boxes_xyxy.shape[0]
        if gt_boxes_xyxy.numel() == 0:
            assigned_gt = torch.full((num_preds,), -1, device=device, dtype=torch.long)
            assigned_labels = torch.full(
                (num_preds,), -1, device=device, dtype=torch.long
            )
            assigned_ious = torch.zeros((num_preds,), device=device)
            return assigned_gt, assigned_labels, assigned_ious

        ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
        cls_prob = pred_cls_logits.sigmoid()
        gt_onehot = torch.nn.functional.one_hot(
            gt_labels, num_classes=pred_cls_logits.shape[1]
        ).to(dtype=cls_prob.dtype)
        pair_cls = cls_prob @ gt_onehot.t()

        metric = (pair_cls.clamp(min=1e-9) ** self.alpha) * (
            ious.clamp(min=1e-9) ** self.beta
        )
        if prior_centers is not None:
            cx = prior_centers[:, 0].unsqueeze(1)
            cy = prior_centers[:, 1].unsqueeze(1)
            in_gt = (cx >= gt_boxes_xyxy[:, 0]) & (cx <= gt_boxes_xyxy[:, 2])
            in_gt = in_gt & (cy >= gt_boxes_xyxy[:, 1]) & (cy <= gt_boxes_xyxy[:, 3])
            has_center = in_gt.any(dim=0)
            if not bool(has_center.all()):
                gt_centers = (gt_boxes_xyxy[:, 0:2] + gt_boxes_xyxy[:, 2:4]) / 2
                diff = prior_centers[:, None, :] - gt_centers[None, :, :]
                dist2 = (diff ** 2).sum(dim=-1)
                missing = ~has_center
                if bool(missing.any()):
                    nearest_idx = dist2[:, missing].argmin(dim=0)
                    in_gt[nearest_idx, missing] = True
            metric = torch.where(in_gt, metric, torch.zeros_like(metric))

        if metric.max() <= 0:
            assigned_gt = torch.full((num_preds,), -1, device=device, dtype=torch.long)
            assigned_labels = torch.full(
                (num_preds,), -1, device=device, dtype=torch.long
            )
            assigned_ious = torch.zeros((num_preds,), device=device)
            return assigned_gt, assigned_labels, assigned_ious

        num_gt = gt_boxes_xyxy.shape[0]
        assigned_gt = torch.full((num_preds,), -1, device=device, dtype=torch.long)
        assigned_ious = torch.zeros((num_preds,), device=device)
        gt_assigned = torch.zeros((num_gt,), device=device, dtype=torch.bool)
        metric_work = metric.clone()

        while True:
            remaining = (~gt_assigned) & (metric_work.max(dim=0).values > 0)
            if not bool(remaining.any()):
                break
            best_scores, best_preds = metric_work[:, remaining].max(dim=0)
            gt_indices = torch.nonzero(remaining, as_tuple=False).squeeze(1)

            unique_preds, inv = torch.unique(best_preds, return_inverse=True)
            for idx, pred_idx in enumerate(unique_preds):
                gt_mask = inv == idx
                if not bool(gt_mask.any()):
                    continue
                best_local = torch.argmax(best_scores[gt_mask])
                gt_global = gt_indices[gt_mask][best_local]
                assigned_gt[pred_idx] = gt_global
                assigned_ious[pred_idx] = ious[pred_idx, gt_global]
                gt_assigned[gt_global] = True
                metric_work[pred_idx, :] = 0

        assigned_labels = torch.full((num_preds,), -1, device=device, dtype=torch.long)
        pos_mask = assigned_gt >= 0
        assigned_labels[pos_mask] = gt_labels[assigned_gt[pos_mask]]
        return assigned_gt, assigned_labels, assigned_ious


class SimOTAAssigner:
    """Simplified Optimal Transport Assignment for anchor-free detection.

    SimOTA dynamically assigns positive samples based on:
    1. Center prior (candidate selection)
    2. Cost matrix computation (VFL + GIoU)
    3. Top-k selection per ground truth

    Args:
        center_radius: Radius (in stride units) for center prior.
        candidate_topk: Number of top candidates to consider for dynamic k.
        iou_weight: Weight for IoU cost in the cost matrix.
        cls_weight: Weight for classification cost in the cost matrix.
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        center_radius: float = 2.5,
        candidate_topk: int = 10,
        iou_weight: float = 6.0,
        cls_weight: float = 1.0,
        num_classes: int = 80,
    ) -> None:
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes

    @torch.no_grad()
    def assign(
        self,
        pred_scores: Tensor,
        priors: Tensor,
        decoded_bboxes: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Perform SimOTA assignment.

        Note: This method runs without gradient computation since assignment
        is not differentiable. This is critical for memory efficiency.

        Args:
            pred_scores: Classification scores after sigmoid activation, of shape
                (num_priors, num_classes). Must be sigmoid scores in [0, 1], NOT
                raw logits. This matches the reference implementation which calls
                `assigner.assign(cls_preds.sigmoid(), ...)`.
            priors: Prior points of shape (num_priors, 4) as [cx, cy, stride_w, stride_h].
            decoded_bboxes: Decoded predicted boxes of shape (num_priors, 4) in xyxy format.
            gt_bboxes: Ground truth boxes of shape (num_gts, 4) in xyxy format.
            gt_labels: Ground truth labels of shape (num_gts,).

        Returns:
            Tuple of:
            - assigned_gt_inds: (num_priors,) with 0 for background, gt_idx+1 for positives.
            - matched_pred_ious: (num_priors,) IoU for positives, 0 for background.
        """
        INF = 1e8
        num_gt = gt_bboxes.size(0)
        num_priors = decoded_bboxes.size(0)

        assigned_gt_inds = decoded_bboxes.new_zeros(num_priors, dtype=torch.long)
        matched_pred_ious = decoded_bboxes.new_zeros(num_priors, dtype=torch.float32)

        if num_gt == 0 or num_priors == 0:
            return assigned_gt_inds, matched_pred_ious

        valid_mask, is_in_boxes_and_centers = self._get_in_gt_and_in_center_info(
            priors, gt_bboxes
        )

        if valid_mask.sum() == 0:
            return assigned_gt_inds, matched_pred_ious

        valid_pred_scores = pred_scores[valid_mask]
        valid_decoded_bboxes = decoded_bboxes[valid_mask]

        pairwise_ious = box_iou(valid_decoded_bboxes, gt_bboxes)
        matched_pred_ious = decoded_bboxes.new_zeros(
            num_priors, dtype=pairwise_ious.dtype
        )

        pairwise_giou = generalized_box_iou(valid_decoded_bboxes, gt_bboxes)
        pairwise_giou_cost = 1.0 - pairwise_giou

        vfl_cost = self._compute_vfl_cost(valid_pred_scores, gt_labels, pairwise_ious)

        cost_matrix = (
            self.cls_weight * vfl_cost
            + self.iou_weight * pairwise_giou_cost
            + (~is_in_boxes_and_centers) * INF
        )

        matched_gt_inds, matched_ious = self._dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt
        )

        valid_indices = torch.where(valid_mask)[0]
        matched_mask = matched_gt_inds >= 0
        if matched_mask.any():
            matched_valid_indices = valid_indices[matched_mask]
            assigned_gt_inds[matched_valid_indices] = matched_gt_inds[matched_mask] + 1
            matched_pred_ious[matched_valid_indices] = matched_ious[matched_mask]

        return assigned_gt_inds, matched_pred_ious

    def _get_in_gt_and_in_center_info(
        self, priors: Tensor, gt_bboxes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Check which priors are inside GT boxes or within center radius.

        Args:
            priors: Prior points of shape (num_priors, 4) as [cx, cy, stride_w, stride_h].
            gt_bboxes: Ground truth boxes of shape (num_gts, 4) in xyxy format.

        Returns:
            Tuple of:
            - valid_mask: (num_priors,) True for priors that are valid candidates.
            - is_in_boxes_and_centers: (num_valid, num_gt) True for valid (prior, gt) pairs.
        """
        prior_cx = priors[:, 0]
        prior_cy = priors[:, 1]
        prior_stride = priors[:, 2]

        is_in_gt_x = (prior_cx[:, None] > gt_bboxes[None, :, 0]) & (
            prior_cx[:, None] < gt_bboxes[None, :, 2]
        )
        is_in_gt_y = (prior_cy[:, None] > gt_bboxes[None, :, 1]) & (
            prior_cy[:, None] < gt_bboxes[None, :, 3]
        )
        is_in_gt = is_in_gt_x & is_in_gt_y  # (num_priors, num_gt)

        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2

        center_radius_pixels = (
            self.center_radius * prior_stride[:, None]
        )  # (num_priors, 1)
        is_in_center_x = (prior_cx[:, None] > gt_cx[None, :] - center_radius_pixels) & (
            prior_cx[:, None] < gt_cx[None, :] + center_radius_pixels
        )
        is_in_center_y = (prior_cy[:, None] > gt_cy[None, :] - center_radius_pixels) & (
            prior_cy[:, None] < gt_cy[None, :] + center_radius_pixels
        )
        is_in_center = is_in_center_x & is_in_center_y  # (num_priors, num_gt)

        is_in_boxes_or_centers = is_in_gt | is_in_center  # (num_priors, num_gt)
        valid_mask = is_in_boxes_or_centers.any(dim=1)  # (num_priors,)

        is_in_boxes_and_centers = is_in_gt & is_in_center  # (num_priors, num_gt)

        is_in_boxes_and_centers = is_in_boxes_and_centers[
            valid_mask
        ]  # (num_valid, num_gt)

        return valid_mask, is_in_boxes_and_centers

    def _compute_vfl_cost(
        self,
        pred_scores: Tensor,
        gt_labels: Tensor,
        pairwise_ious: Tensor,
    ) -> Tensor:
        """Compute VFL-based classification cost.

        Uses F.binary_cross_entropy (not with_logits) since pred_scores are
        already sigmoid-activated.

        Args:
            pred_scores: Sigmoid scores of shape (num_valid, num_classes).
            gt_labels: GT class indices of shape (num_gt,).
            pairwise_ious: IoU matrix of shape (num_valid, num_gt).

        Returns:
            Cost matrix of shape (num_valid, num_gt).
        """
        alpha = 0.75
        gamma = 2.0
        num_valid = pred_scores.size(0)
        num_gt = gt_labels.size(0)

        # Build VFL score matrix: (num_valid * num_gt, num_classes)
        gt_vfl_labels = gt_labels.unsqueeze(0).expand(num_valid, -1).reshape(-1)
        pred_scores_expanded = (
            pred_scores.unsqueeze(1)
            .expand(-1, num_gt, -1)
            .reshape(-1, self.num_classes)
        )

        # VFL target: zeros everywhere except IoU at the GT class
        vfl_target = pairwise_ious.new_zeros(pred_scores_expanded.shape)
        vfl_target[
            torch.arange(vfl_target.size(0), device=vfl_target.device), gt_vfl_labels
        ] = pairwise_ious.reshape(-1)

        # VFL focal weight: IoU for positives, focal modulation for negatives
        focal_weight = (
            vfl_target * (vfl_target > 0).float()
            + alpha
            * (pred_scores_expanded - vfl_target).abs().pow(gamma)
            * (vfl_target <= 0).float()
        )

        # BCE (not with_logits since pred_scores are already sigmoid)
        # Clamp to avoid log(0)
        pred_clamped = pred_scores_expanded.clamp(min=1e-7, max=1 - 1e-7)
        bce = -(
            vfl_target * pred_clamped.log()
            + (1 - vfl_target) * (1 - pred_clamped).log()
        )
        losses_vfl = (bce * focal_weight).sum(dim=1)  # Sum over classes

        return torch.reshape(losses_vfl, (num_valid, num_gt))

    def _dynamic_k_matching(
        self,
        cost_matrix: Tensor,
        pairwise_ious: Tensor,
        num_gt: int,
    ) -> tuple[Tensor, Tensor]:
        """Perform dynamic k matching for SimOTA.

        Args:
            cost_matrix: Cost matrix of shape (num_valid, num_gt).
            pairwise_ious: IoU matrix of shape (num_valid, num_gt).
            num_gt: Number of ground truth boxes.

        Returns:
            Tuple of:
            - matched_gt_inds: (num_valid,) matched GT index, -1 for unmatched.
            - matched_ious: (num_valid,) IoU with matched GT, 0 for unmatched.
        """
        num_valid = cost_matrix.size(0)
        device = cost_matrix.device
        matching_matrix = torch.zeros_like(cost_matrix)

        n_candidate = min(self.candidate_topk, num_valid)
        topk_ious, _ = pairwise_ious.topk(n_candidate, dim=0, largest=True)
        dynamic_ks = topk_ious.sum(dim=0).clamp(min=1).int()  # (num_gt,)

        for gt_idx in range(num_gt):
            k = int(dynamic_ks[gt_idx].item())
            _, pos_idx = cost_matrix[:, gt_idx].topk(k, largest=False)
            matching_matrix[pos_idx, gt_idx] = 1

        matched_per_prior = matching_matrix.sum(dim=1)
        conflict_mask = matched_per_prior > 1
        has_conflicts = conflict_mask.any()
        if has_conflicts:
            conflict_costs = cost_matrix.clone()
            conflict_costs[matching_matrix == 0] = float("inf")
            best_gt_per_prior = conflict_costs.argmin(dim=1)

            conflict_indices = torch.where(conflict_mask)[0]
            matching_matrix[conflict_indices, :] = 0
            matching_matrix[conflict_indices, best_gt_per_prior[conflict_indices]] = 1

        matched_gt_inds = torch.full((num_valid,), -1, dtype=torch.long, device=device)
        matched_ious = pairwise_ious.new_zeros(num_valid)

        matched_mask = matching_matrix.sum(dim=1) > 0
        if matched_mask.any():
            matched_gt_inds_for_matched = matching_matrix[matched_mask].argmax(dim=1)
            matched_gt_inds[matched_mask] = matched_gt_inds_for_matched

            matched_indices = torch.where(matched_mask)[0]
            matched_ious[matched_mask] = pairwise_ious[
                matched_indices, matched_gt_inds_for_matched
            ]

        return matched_gt_inds, matched_ious
