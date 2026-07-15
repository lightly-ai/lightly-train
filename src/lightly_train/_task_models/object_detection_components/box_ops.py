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
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area

_logger = logging.getLogger(__name__)
_invalid_bbox_warning_emitted = False


def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def sanitize_boxes_cxcywh_normalized(
    boxes: Tensor,
    image_size: tuple[int, int] | None = None,
    min_size_px: float = 0.0,
) -> Tensor:
    """Sanitize normalized ``cxcywh`` boxes predicted by the decoder.

    The RT-DETR matcher and criterion operate on boxes normalized by image
    size. We therefore map NaN to ``0``, ``-inf`` to ``0``, ``+inf`` to ``1``,
    and clamp all coordinates to ``[0, 1]`` before downstream L1/IoU/GIoU
    computations.

    When ``image_size`` (H, W) and a positive ``min_size_px`` are provided,
    the per-side width and height are additionally clamped up to at least
    ``min_size_px`` pixels. This is the LT-DETR training-codepath guard that
    keeps degenerate boxes from destabilizing the matcher and losses.
    """
    global _invalid_bbox_warning_emitted

    if not _invalid_bbox_warning_emitted:
        invalid_mask = ~torch.isfinite(boxes)
        if invalid_mask.any():
            _logger.warning(
                "Found invalid predicted bbox values (NaN/inf) before "
                "sanitization. This usually indicates numerical instability "
                "upstream."
            )
            _invalid_bbox_warning_emitted = True

    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    boxes = boxes.clamp(min=0.0, max=1.0)

    if image_size is not None and min_size_px > 0.0:
        height, width = image_size
        if height > 0 and width > 0:
            min_w = min(float(min_size_px) / float(width), 1.0)
            min_h = min(float(min_size_px) / float(height), 1.0)
            # Clamp width and height up to the per-side minimum. Build a fresh
            # tensor rather than mutating ``boxes`` in-place because callers pass
            # slices of autograd-tracked prediction tensors whose storage must
            # stay intact for backward.
            boxes = torch.stack(
                [
                    boxes[..., 0],
                    boxes[..., 1],
                    boxes[..., 2].clamp(min=min_w),
                    boxes[..., 3].clamp(min=min_h),
                ],
                dim=-1,
            )

    return boxes


# modified from torchvision to also return the union
def box_iou(boxes1: Tensor, boxes2: Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
