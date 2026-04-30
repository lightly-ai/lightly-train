#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from PIL.Image import Image as PILImage
from PIL.ImageDraw import ImageDraw as PILDraw
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize.utils import (
    _cxcywh_to_xyxy,
    _denormalize_image,
    _draw_bbox_label,
    _get_class_color,
    _render_grid,
)
from lightly_train.types import ObjectDetectionBatch


def plot_object_detection_labels(
    batch: ObjectDetectionBatch,
    included_classes: dict[int, str],
    max_images: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PILImage:
    """Render a grid of images annotated with ground truth bounding boxes.

    Args:
        batch: Object detection batch with images, bboxes (cxcywh normalized), and
            classes.
        included_classes: Mapping from class ID to class name.
        mean: Per-channel mean used for image normalization (for denormalization).
        std: Per-channel std used for image normalization (for denormalization).
        max_images: Maximum number of images to include in the grid.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    gt_images = batch["image"].cpu()
    gt_bboxes = [b.cpu() for b in batch["bboxes"]]
    gt_classes = [c.cpu() for c in batch["classes"]]
    n = min(max_images, gt_images.shape[0])

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone().to(dtype=torch.float32)
        if mean is not None and std is not None:
            image_tensor = _denormalize_image(image=image_tensor, mean=mean, std=std)

        _, img_height, img_width = image_tensor.shape
        img = torchvision_functional.to_pil_image(image_tensor)
        draw = PILDraw(img)

        boxes = gt_bboxes[i]
        class_ids = gt_classes[i]
        if len(boxes) > 0:
            boxes_xyxy = _cxcywh_to_xyxy(boxes=boxes, w=img_width, h=img_height)
            for box, class_id in zip(boxes_xyxy, class_ids):
                x1, y1, x2, y2 = box.tolist()
                class_name = included_classes.get(
                    int(class_id), f"Class {int(class_id)}"
                )
                color = _get_class_color(int(class_id))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                _draw_bbox_label(
                    draw=draw,
                    x1=x1,
                    y1=y1,
                    text=class_name,
                    color=color,
                )
        pil_images.append(img)

    return _render_grid(pil_images)


def plot_object_detection_predictions(
    batch: ObjectDetectionBatch,
    results: list[dict[str, Tensor]],
    included_classes: dict[int, str],
    max_images: int,
    score_threshold: float,
    max_pred_boxes: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PILImage:
    """Render a grid of images annotated with predicted bounding boxes.

    Predictions are filtered by score_threshold and capped at max_pred_boxes per
    image, selecting the highest-confidence detections.

    Args:
        batch: Object detection batch with images and original_size.
        results: Postprocessor outputs, each a dict with 'boxes' (xyxy in original
            image coordinates), 'labels', and 'scores'.
        included_classes: Mapping from class ID to class name.
        score_threshold: Minimum score for a predicted box to be shown.
        max_pred_boxes: Maximum number of predicted boxes to show per image.
        mean: Per-channel mean used for image normalization (for denormalization).
        std: Per-channel std used for image normalization (for denormalization).

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    results = [
        {k: v.cpu() if isinstance(v, Tensor) else v for k, v in r.items()}
        for r in results
    ]
    gt_images = batch["image"].cpu()
    orig_target_sizes = batch["original_size"]
    n = min(max_images, gt_images.shape[0])

    _, img_height, img_width = gt_images.shape[1:]

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone().to(dtype=torch.float32)
        if mean is not None and std is not None:
            image_tensor = _denormalize_image(
                image=image_tensor,
                mean=mean,
                std=std,
            )

        img = torchvision_functional.to_pil_image(image_tensor)
        draw = PILDraw(img)

        result = results[i]
        boxes = result["boxes"]
        class_ids = result["labels"]
        scores = result["scores"]

        if len(boxes) > 0:
            # Scale predicted boxes from original image coordinates to tensor dimensions.
            orig_width, orig_height = orig_target_sizes[i]
            boxes = boxes.clone()
            boxes[:, 0] = boxes[:, 0] * img_width / orig_width
            boxes[:, 1] = boxes[:, 1] * img_height / orig_height
            boxes[:, 2] = boxes[:, 2] * img_width / orig_width
            boxes[:, 3] = boxes[:, 3] * img_height / orig_height

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
                        int(class_id), f"Class {int(class_id)}"
                    )
                    color = _get_class_color(int(class_id))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    _draw_bbox_label(
                        draw=draw,
                        x1=x1,
                        y1=y1,
                        text=f"{class_name} {score:.2f}",
                        color=color,
                    )
        pil_images.append(img)

    return _render_grid(pil_images)
