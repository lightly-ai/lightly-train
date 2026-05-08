#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize.utils import (
    _bboxes_from_masks,
    _build_instance_mask_overlay,
    _cxcywh_to_xyxy,
    _denormalize_image,
    _draw_labeled_boxes,
    _render_grid,
)
from lightly_train.types import InstanceSegmentationBatch


def plot_instance_segmentation_labels(
    batch: InstanceSegmentationBatch,
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with ground truth instance segmentation masks.

    Each instance is overlaid in the color of its class.
    The class name is rendered as a label at the top-left corner of each instance's bounding box.

    Args:
        batch: Instance segmentation batch with images, per-image binary masks
            (one mask per instance) and bboxes (cxcywh normalized). Mask labels
            are internal contiguous class indices.
        class_names: A dict mapping internal class IDs to class names.
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

        masks = binary_masks_list[i]["masks"].cpu().bool()
        labels = binary_masks_list[i]["labels"].cpu()
        bboxes_norm = bboxes_list[i].cpu()

        overlay = _build_instance_mask_overlay(
            masks=masks, labels=labels, size=img.size, class_names=class_names
        )
        blended = Image.blend(img, overlay, alpha=alpha)
        bboxes_xyxy = _cxcywh_to_xyxy(boxes=bboxes_norm, w=img.size[0], h=img.size[1])
        _draw_labeled_boxes(
            image=blended,
            bboxes_xyxy=bboxes_xyxy,
            labels=labels,
            scores=None,
            class_names=class_names,
        )
        pil_images.append(blended)

    return _render_grid(pil_images)


def plot_instance_segmentation_predictions(
    batch: InstanceSegmentationBatch,
    predictions: list[dict[str, Tensor]],
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
    score_threshold: float,
) -> PILImage:
    """Render a grid of images annotated with predicted instance segmentation masks.

    Each instance is overlaid in the color of its class. The class name and score
    are rendered as a label at the top-left corner of each instance's bounding box,
    where the bounding box is derived from the predicted mask. Predictions with a
    score below score_threshold are filtered out.

    Args:
        batch: Instance segmentation batch with images.
        predictions: A list of per-image dicts with "labels" of shape (Q,),
            "masks" of shape (Q, H, W) and "scores" of shape (Q,). Labels are
            internal contiguous class indices.
        class_names: A dict mapping internal class IDs to class names.
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.
        alpha: Blending factor for the mask overlay in [0, 1]. 0 shows only the
            image, 1 shows only the mask.
        score_threshold: Minimum score for a predicted instance to be shown.

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

        masks = predictions[i]["masks"].cpu().bool()
        labels = predictions[i]["labels"].cpu()
        scores = predictions[i]["scores"].cpu()

        keep_scores = scores >= score_threshold
        masks = masks[keep_scores]
        labels = labels[keep_scores]
        scores = scores[keep_scores]

        overlay = _build_instance_mask_overlay(
            masks=masks, labels=labels, size=img.size, class_names=class_names
        )
        blended = Image.blend(img, overlay, alpha=alpha)

        bboxes_xyxy, keep = _bboxes_from_masks(masks=masks)
        _draw_labeled_boxes(
            image=blended,
            bboxes_xyxy=bboxes_xyxy,
            labels=labels[keep],
            scores=scores[keep],
            class_names=class_names,
        )
        pil_images.append(blended)

    return _render_grid(pil_images)
