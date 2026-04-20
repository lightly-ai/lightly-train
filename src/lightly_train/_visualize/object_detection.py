#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize.utils import (
    denormalize_image,
    draw_label,
    get_class_color,
    load_font,
)


def _cxcywh_to_xyxy(boxes: Tensor, w: int, h: int) -> Tensor:
    """Convert bounding boxes from cxcywh format to xyxy format.

    Args:
        boxes: Tensor of shape (n_boxes, 4) in cxcywh format (center_x, center_y,
            width, height). Values are normalized to [0, 1].
        w: Width of the image.
        h: Height of the image.

    Returns:
        Tensor of shape (n_boxes, 4) in xyxy format (x1, y1, x2, y2).
    """
    boxes_xyxy = boxes.clone()
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes_xyxy[:, 0] = (cx - bw / 2) * w
    boxes_xyxy[:, 1] = (cy - bh / 2) * h
    boxes_xyxy[:, 2] = (cx + bw / 2) * w
    boxes_xyxy[:, 3] = (cy + bh / 2) * h
    return boxes_xyxy


def plot_object_detection_comparison(
    images: Tensor,
    gt_bboxes: list[Tensor],
    gt_classes: list[Tensor],
    pred_bboxes: list[Tensor],
    pred_classes: list[Tensor],
    pred_scores: list[Tensor],
    included_classes: dict[int, str],
    score_threshold: float = 0.3,
    max_pred_boxes: int = 32,
    max_images: int | None = None,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> list[Image.Image]:
    """Plot side-by-side comparison of ground truth labels and predictions.

    For each image, the left side shows all ground truth boxes and the right side
    shows predictions filtered by score threshold and limited to the top boxes by
    confidence score.

    Args:
        images: Batch of images with shape (batch_size, 3, H, W).
        gt_bboxes: List of ground truth bounding box tensors. Each tensor has shape
            (n_boxes, 4) with coordinates in cxcywh format normalized to [0, 1].
        gt_classes: List of ground truth class label tensors. Each tensor has shape
            (n_boxes,).
        pred_bboxes: List of predicted bounding box tensors. Each tensor has shape
            (n_boxes, 4) with coordinates in xyxy format scaled to image tensor
            dimensions.
        pred_classes: List of predicted class label tensors. Each tensor has shape
            (n_boxes,).
        pred_scores: List of confidence score tensors. Each tensor has shape
            (n_boxes,).
        included_classes: Dictionary mapping class IDs to class names.
        score_threshold: Minimum confidence score to draw a prediction box.
        max_pred_boxes: Maximum number of prediction boxes to draw per image,
            selected by highest confidence score.
        max_images: Maximum number of images to process. If None, processes all.
        mean: Tuple of mean values for denormalization.
        std: Tuple of std values for denormalization.

    Returns:
        List of PIL images, each showing GT on the left and predictions on the right.
    """

    pil_images = []
    batch_size = images.shape[0]
    n = min(max_images or batch_size, batch_size)

    for i in range(n):
        image_tensor = images[i].clone()
        if mean is not None and std is not None:
            image_tensor = denormalize_image(image_tensor, mean, std)

        _, img_height, img_width = image_tensor.shape

        # --- Ground truth side (left) ---
        gt_image = torchvision_functional.to_pil_image(image_tensor)
        gt_draw = ImageDraw.Draw(gt_image)
        font = load_font(size=18)

        boxes = gt_bboxes[i]
        class_ids = gt_classes[i]

        if len(boxes) > 0:
            boxes_xyxy = _cxcywh_to_xyxy(boxes, img_width, img_height)
            for box, class_id in zip(boxes_xyxy, class_ids):
                x1, y1, x2, y2 = box.tolist()
                class_name = included_classes.get(int(class_id), f"Class {class_id}")
                color = get_class_color(int(class_id))
                gt_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw_label(gt_draw, x1, y1, class_name, color, font)

        # --- Prediction side (right) ---
        pred_image = torchvision_functional.to_pil_image(image_tensor)
        pred_draw = ImageDraw.Draw(pred_image)

        boxes = pred_bboxes[i]
        class_ids = pred_classes[i]
        scores = pred_scores[i]

        if len(boxes) > 0:
            # Filter by threshold, then sort by score descending, take top N
            mask = scores >= score_threshold
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            scores = scores[mask]

            if len(scores) > 0:
                order = torch.argsort(scores, descending=True)[:max_pred_boxes]
                boxes = boxes[order]
                class_ids = class_ids[order]
                scores = scores[order]

                for box, class_id, score in zip(boxes, class_ids, scores):
                    x1, y1, x2, y2 = box.tolist()
                    class_name = included_classes.get(
                        int(class_id), f"Class {class_id}"
                    )
                    color = get_class_color(int(class_id))
                    pred_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw_label(
                        pred_draw, x1, y1, f"{class_name} {score:.2f}", color, font
                    )

        # Combine side by side
        combined_width = gt_image.width + pred_image.width
        combined = Image.new("RGB", (combined_width, gt_image.height))
        combined.paste(gt_image, (0, 0))
        combined.paste(pred_image, (gt_image.width, 0))

        # Resize the height to 320 while maintaining aspect ratio (to save space)
        # aspect_ratio = combined.width / combined.height
        # new_height = 320
        # new_width = int(aspect_ratio * new_height)
        # combined = combined.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

        pil_images.append(combined)
    return pil_images
