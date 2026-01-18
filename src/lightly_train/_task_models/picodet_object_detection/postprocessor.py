#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import batched_nms

from lightly_train._task_models.picodet_object_detection.pico_head import (
    Integral,
    distance2bbox,
)


class PicoDetPostProcessor(nn.Module):
    """Post-processing for PicoDet predictions.

    This module decodes the raw predictions from PicoHead into bounding boxes
    and applies NMS filtering.

    Args:
        num_classes: Number of object classes.
        reg_max: Maximum value for DFL distribution.
        strides: Stride for each feature level.
        score_threshold: Minimum score threshold for detections.
        iou_threshold: IoU threshold for NMS.
        max_detections: Maximum number of detections to return.
        nms_pre: Maximum number of candidates per level before NMS.
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 7,
        strides: Sequence[int] = (8, 16, 32, 64),
        score_threshold: float = 0.025,
        iou_threshold: float = 0.6,
        max_detections: int = 100,
        nms_pre: int = 1000,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.nms_pre = nms_pre

        self.integral = Integral(reg_max)
        self.deploy_mode = False

    def deploy(self) -> None:
        """Set deploy mode for inference."""
        self.deploy_mode = True

    def _generate_grid_points(
        self, height: int, width: int, stride: int, device: torch.device
    ) -> Tensor:
        """Generate grid center points for a feature map.

        Args:
            height: Feature map height.
            width: Feature map width.
            stride: Stride of the feature map.
            device: Device to create tensors on.

        Returns:
            Grid points of shape (H*W, 2) as [x, y] in pixel coordinates.
        """
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def forward(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        original_size: tuple[int, int],
        score_threshold: float | None = None,
    ) -> dict[str, Tensor]:
        """Process predictions and return final detections.

        This is designed for single-image inference (batch size 1).

        Args:
            cls_scores: List of classification scores per level, each (1, C, H, W).
            bbox_preds: List of bbox distributions per level, each (1, 4*(reg_max+1), H, W).
            original_size: Original image size (height, width) for scaling boxes.
            score_threshold: Optional override for score threshold.

        Returns:
            Dictionary with:
            - labels: Tensor of shape (N,) with class indices.
            - bboxes: Tensor of shape (N, 4) with boxes in xyxy format.
            - scores: Tensor of shape (N,) with confidence scores.
        """
        score_thr = self.score_threshold if score_threshold is None else score_threshold

        assert len(cls_scores) == len(bbox_preds) == len(self.strides)
        assert cls_scores[0].shape[0] == 1, "Only batch size 1 is supported"

        device = cls_scores[0].device
        orig_h, orig_w = original_size

        in_h = cls_scores[0].shape[-2] * self.strides[0]
        in_w = cls_scores[0].shape[-1] * self.strides[0]

        per_level_boxes: list[Tensor] = []
        per_level_scores: list[Tensor] = []
        per_level_labels: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[level_idx]
            _, num_classes, height, width = cls_score.shape

            cls_score = cls_score[0].permute(1, 2, 0).reshape(-1, num_classes)
            bbox_pred = (
                bbox_pred[0].permute(1, 2, 0).reshape(-1, 4 * (self.reg_max + 1))
            )

            points = self._generate_grid_points(height, width, stride, device)

            scores = cls_score.sigmoid().reshape(-1, num_classes)
            valid_mask = scores > score_thr
            if not valid_mask.any():
                per_level_boxes.append(points.new_zeros((0, 4)))
                per_level_scores.append(points.new_zeros((0,)))
                per_level_labels.append(points.new_zeros((0,), dtype=torch.long))
                continue

            scores_valid = scores[valid_mask]
            valid_idxs = valid_mask.nonzero(as_tuple=False)
            num_topk = (
                valid_idxs.size(0)
                if self.nms_pre <= 0
                else min(self.nms_pre, valid_idxs.size(0))
            )
            scores_sorted, idxs = scores_valid.sort(descending=True)
            scores = scores_sorted[:num_topk]
            topk_idxs = valid_idxs[idxs[:num_topk]]
            keep_idxs = topk_idxs[:, 0]
            labels = topk_idxs[:, 1]

            bbox_pred = bbox_pred[keep_idxs]
            points = points[keep_idxs]
            distances = self.integral(bbox_pred) * stride
            boxes = distance2bbox(points, distances)

            per_level_boxes.append(boxes)
            per_level_scores.append(scores)
            per_level_labels.append(labels)

        boxes = torch.cat(per_level_boxes, dim=0)
        scores = torch.cat(per_level_scores, dim=0)
        labels = torch.cat(per_level_labels, dim=0)

        if self.deploy_mode:
            scores_sorted, idxs = scores.sort(descending=True)
            idxs = idxs[: self.max_detections]
            boxes = boxes[idxs]
            labels = labels[idxs]
            scores = scores_sorted[: self.max_detections]
        else:
            keep = batched_nms(boxes, scores, labels, self.iou_threshold)
            keep = keep[: self.max_detections]
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        scale_x = orig_w / float(in_w)
        scale_y = orig_h / float(in_h)
        boxes = boxes * boxes.new_tensor([scale_x, scale_y, scale_x, scale_y])

        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    def forward_batch(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        original_sizes: Tensor,
        score_threshold: float | None = None,
    ) -> list[dict[str, Tensor]]:
        """Process predictions for a batch of images.

        Args:
            cls_scores: List of classification scores per level, each (B, C, H, W).
            bbox_preds: List of bbox distributions per level, each (B, 4*(reg_max+1), H, W).
            original_sizes: Original image sizes of shape (B, 2) as [height, width].
            score_threshold: Optional override for score threshold.

        Returns:
            List of dictionaries (one per image), each containing:
            - labels: Tensor of shape (N,) with class indices.
            - bboxes: Tensor of shape (N, 4) with boxes in xyxy format.
            - scores: Tensor of shape (N,) with confidence scores.
        """
        batch_size = cls_scores[0].shape[0]
        results = []

        for batch_idx in range(batch_size):
            single_cls_scores = [cs[batch_idx : batch_idx + 1] for cs in cls_scores]
            single_bbox_preds = [bp[batch_idx : batch_idx + 1] for bp in bbox_preds]
            orig_size = (
                int(original_sizes[batch_idx, 0].item()),
                int(original_sizes[batch_idx, 1].item()),
            )

            result = self.forward(
                single_cls_scores,
                single_bbox_preds,
                orig_size,
                score_threshold,
            )
            results.append(result)

        return results
