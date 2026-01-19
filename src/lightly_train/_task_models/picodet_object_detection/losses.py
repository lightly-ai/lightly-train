#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VarifocalLoss(nn.Module):
    """Varifocal Loss for IoU-aware classification.

    This loss is used in PicoDet to train the classification branch with
    IoU-aware soft targets. Positive samples are weighted by their IoU
    with ground truth, while negative samples use focal loss weighting.

    Reference: VarifocalNet: An IoU-aware Dense Object Detector (CVPR 2021)

    Args:
        alpha: Weighting factor for negative samples.
        gamma: Focusing parameter for focal modulation.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: Tensor, target: Tensor) -> Tensor:
        """Compute varifocal loss.

        Args:
            pred_logits: Predicted logits of shape (N, C).
            target: Target values of shape (N, C) where positive positions
                contain IoU values and negative positions are 0.

        Returns:
            Scalar loss value (summed, not averaged).
        """
        pred_sigmoid = pred_logits.sigmoid()
        target = target.to(dtype=pred_sigmoid.dtype)

        # Focal weight:
        # - For positives (target > 0): weight = target (IoU value)
        # - For negatives (target == 0): weight = alpha * |pred - target|^gamma
        focal_weight = target * (target > 0).to(target.dtype) + (
            self.alpha
            * (pred_sigmoid - target).abs().pow(self.gamma)
            * (target <= 0).to(target.dtype)
        )

        # Binary cross-entropy with focal weighting
        loss = (
            F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
            * focal_weight
        )
        return loss.sum()


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss for bounding box regression.

    This loss supervises the distribution P(x) where x ∈ {0, 1, ..., reg_max}.
    It uses soft labels between adjacent bins for non-integer targets.

    Reference: Generalized Focal Loss (NeurIPS 2020)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: Tensor, target: Tensor, weight: Tensor | None = None
    ) -> Tensor:
        """Compute distribution focal loss.

        Args:
            pred: Distribution logits of shape (N, 4*(reg_max+1)) or (N, reg_max+1).
            target: Continuous distance targets of shape (N, 4) or (N,) in [0, reg_max].
            weight: Optional per-element weights of shape (N*4,) or (N,).

        Returns:
            Scalar loss value (sum if weighted, mean otherwise).
        """
        # Infer reg_max from pred shape
        if pred.dim() == 2 and target.dim() == 2:
            # (N, 4*(reg_max+1)) and (N, 4)
            reg_max = pred.shape[-1] // 4 - 1
            pred = pred.reshape(-1, reg_max + 1)
            target = target.reshape(-1)
        elif pred.dim() == 2 and target.dim() == 1:
            # Already flattened
            reg_max = pred.shape[-1] - 1
        else:
            raise ValueError(
                f"Unexpected shapes: pred={pred.shape}, target={target.shape}"
            )

        target = target.clamp(0, reg_max - 0.01)

        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()

        loss_left = F.cross_entropy(pred, target_left, reduction="none") * weight_left
        loss_right = (
            F.cross_entropy(pred, target_right, reduction="none") * weight_right
        )

        loss = loss_left + loss_right

        if weight is not None:
            loss = loss * weight
            return loss.sum()

        return loss.mean()


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: Boxes of shape (N, 4) in xyxy format.
        boxes2: Boxes of shape (M, 4) in xyxy format.

    Returns:
        IoU matrix of shape (N, M).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter

    return inter / union.clamp(min=1e-6)


def box_iou_aligned(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute IoU between aligned pairs of boxes.

    Args:
        boxes1: Boxes of shape (N, 4) in xyxy format.
        boxes2: Boxes of shape (N, 4) in xyxy format.

    Returns:
        IoU values of shape (N,).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter

    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute Generalized IoU between two sets of boxes.

    Args:
        boxes1: Boxes of shape (N, 4) in xyxy format.
        boxes2: Boxes of shape (M, 4) in xyxy format.

    Returns:
        GIoU matrix of shape (N, M).
    """
    # Regular IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


def generalized_box_iou_aligned(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute Generalized IoU between aligned pairs of boxes.

    Args:
        boxes1: Boxes of shape (N, 4) in xyxy format.
        boxes2: Boxes of shape (N, 4) in xyxy format.

    Returns:
        GIoU values of shape (N,).
    """
    # Regular IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, 0] * wh_enc[:, 1]

    return iou - (area_enc - union) / area_enc.clamp(min=1e-6)


class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression.

    Reference: Generalized Intersection over Union (CVPR 2019)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred_boxes: Tensor,
        target_boxes: Tensor,
        weight: Tensor | None = None,
    ) -> Tensor:
        """Compute GIoU loss.

        Args:
            pred_boxes: Predicted boxes of shape (N, 4) in xyxy format.
            target_boxes: Target boxes of shape (N, 4) in xyxy format.
            weight: Optional per-box weights of shape (N,).

        Returns:
            Scalar loss value.
        """
        giou = generalized_box_iou_aligned(pred_boxes, target_boxes)
        loss = 1 - giou

        if weight is not None:
            loss = loss * weight
            result: Tensor = loss.sum()
            return result

        result = loss.mean()
        return result
