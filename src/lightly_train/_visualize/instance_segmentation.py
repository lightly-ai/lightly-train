#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from PIL import Image
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
from lightly_train.types import InstanceSegmentationBatch


def plot_instance_segmentation_labels(
    batch: InstanceSegmentationBatch,
    included_classes: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float = 0.6,
) -> PILImage:
    """Render a grid of images annotated with ground truth instance segmentation masks.

    Each instance is overlaid in the color of its class and outlined with a thin
    contour that separates it from the background and from neighboring instances
    of the same class. The class name is rendered as a label at the top-left
    corner of each instance's bounding box.

    Args:
        batch: Instance segmentation batch with images, per-image binary masks
            (one mask per instance) and bboxes (cxcywh normalized). Mask labels
            are internal contiguous class indices.
        included_classes: A dict mapping internal class IDs to class names.
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.
        alpha: Blending factor for the mask overlay in [0, 1]. 0 shows only the
            image, 1 shows only the mask.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    images = batch["image"]
    binary_masks_list = batch["binary_masks"]
    bboxes_list = batch["bboxes"]
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    n = min(max_images, len(gt_images))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if image_normalize is not None:
            image_tensor = _denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )
        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        masks = binary_masks_list[i]["masks"].cpu()
        labels = binary_masks_list[i]["labels"].cpu()
        bboxes = bboxes_list[i].cpu()

        class_mask, instance_id_mask = _build_class_and_instance_id_masks(
            masks=masks, labels=labels
        )
        overlay = _build_class_overlay(
            class_mask=class_mask, size=img.size, class_names=included_classes
        )
        blended = Image.blend(img, overlay, alpha=alpha)

        if len(bboxes) > 0:
            img_width, img_height = blended.size
            bboxes_xyxy = _cxcywh_to_xyxy(boxes=bboxes, w=img_width, h=img_height)
            draw = PILDraw(blended)
            for box, class_id in zip(bboxes_xyxy, labels):
                x1, y1, x2, y2 = box.tolist()
                class_id_int = int(class_id)
                class_name = included_classes.get(
                    class_id_int, f"Class {class_id_int}"
                )
                color = _get_class_color(class_id_int)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                _draw_bbox_label(
                    draw=draw,
                    x1=x1,
                    y1=y1,
                    text=class_name,
                    color=color,
                )

        pil_images.append(blended)

    return _render_grid(pil_images)


def plot_instance_segmentation_predictions(
    batch: InstanceSegmentationBatch,
    predictions: list[dict[str, Tensor]],
    included_classes: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    score_threshold: float = 0.5,
    max_predictions: int = 100,
    alpha: float = 0.6,
) -> PILImage:
    """Render a grid of images annotated with predicted instance segmentation masks.

    Predictions are filtered by score_threshold and capped at max_predictions per
    image, keeping the highest-confidence instances. Each kept instance is
    overlaid in the color of its class and outlined with a bounding box derived
    from the mask. The class name and score are rendered as a label at the
    top-left corner of each instance's bounding box.

    Args:
        batch: Instance segmentation batch with images.
        predictions: A list of per-image dicts with "labels" of shape (Q,),
            "masks" of shape (Q, H, W) and "scores" of shape (Q,). Labels are
            internal contiguous class indices.
        included_classes: A dict mapping internal class IDs to class names.
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.
        score_threshold: Minimum score for a predicted instance to be shown.
        max_predictions: Maximum number of predicted instances to show per image.
        alpha: Blending factor for the mask overlay in [0, 1]. 0 shows only the
            image, 1 shows only the mask.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    images = batch["image"]
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    n = min(max_images, len(gt_images))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if image_normalize is not None:
            image_tensor = _denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )
        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")

        pred = predictions[i]
        masks = pred["masks"].cpu()
        labels = pred["labels"].cpu()
        scores = pred["scores"].cpu()

        keep = scores >= score_threshold
        masks = masks[keep]
        labels = labels[keep]
        scores = scores[keep]
        if scores.numel() > max_predictions:
            order = torch.argsort(scores, descending=True)[:max_predictions]
            masks = masks[order]
            labels = labels[order]
            scores = scores[order]

        class_mask, _ = _build_class_and_instance_id_masks(
            masks=masks.bool(), labels=labels
        )
        overlay = _build_class_overlay(
            class_mask=class_mask, size=img.size, class_names=included_classes
        )
        blended = Image.blend(img, overlay, alpha=alpha)

        if masks.shape[0] > 0:
            draw = PILDraw(blended)
            for mask, class_id, score in zip(masks, labels, scores):
                ys, xs = torch.where(mask)
                if ys.numel() == 0:
                    continue
                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max())
                y2 = int(ys.max())
                class_id_int = int(class_id)
                class_name = included_classes.get(
                    class_id_int, f"Class {class_id_int}"
                )
                color = _get_class_color(class_id_int)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                _draw_bbox_label(
                    draw=draw,
                    x1=x1,
                    y1=y1,
                    text=f"{class_name} {float(score):.2f}",
                    color=color,
                )

        pil_images.append(blended)

    return _render_grid(pil_images)


def _build_class_and_instance_id_masks(
    masks: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build 2D class id and instance id masks from per-instance binary masks.

    Larger instances are painted first so that smaller, nested instances remain
    visible on top. Background pixels keep a sentinel value of -1 in both
    masks; this value is never a valid class id or instance index, so it stays
    transparent in the overlay and produces clean boundaries against any
    foreground.

    Args:
        masks: Boolean tensor of shape (n_instances, H, W).
        labels: Tensor of shape (n_instances,) with internal class ids.

    Returns:
        A (class_mask, instance_id_mask) tuple, each a tensor of shape (H, W).
    """
    h, w = masks.shape[-2:]
    class_mask = torch.full((h, w), -1, dtype=torch.long)
    instance_id_mask = torch.full((h, w), -1, dtype=torch.long)
    if masks.shape[0] == 0:
        return class_mask, instance_id_mask

    areas = masks.flatten(1).sum(dim=1)
    order = torch.argsort(areas, descending=True)
    for idx in order.tolist():
        mask = masks[idx]
        class_mask[mask] = int(labels[idx])
        instance_id_mask[mask] = idx
    return class_mask, instance_id_mask


def _build_class_overlay(
    class_mask: Tensor,
    size: tuple[int, int],
    class_names: dict[int, str],
) -> PILImage:
    """Build an RGB overlay image where each pixel is colored by its class id.

    Only ids that appear as keys of ``class_names`` are colored; other ids are
    left black.

    Args:
        class_mask: Tensor of shape (H, W) with internal class ids and -1 for
            background.
        size: Target (width, height) of the overlay.
        class_names: Mapping from class id to class name.

    Returns:
        RGB PIL image of the requested size.
    """
    h, w = class_mask.shape[-2:]
    overlay = torch.zeros((3, h, w), dtype=torch.uint8)
    for class_id in torch.unique(class_mask).tolist():
        class_id = int(class_id)
        if class_id not in class_names:
            continue
        color = _get_class_color(class_id)
        class_pixels = class_mask == class_id
        for c in range(3):
            overlay[c][class_pixels] = color[c]

    overlay_img: PILImage = Image.fromarray(overlay.permute(1, 2, 0).numpy()).convert(
        "RGB"
    )
    if overlay_img.size != size:
        overlay_img = overlay_img.resize(size, resample=Image.Resampling.NEAREST)
    return overlay_img