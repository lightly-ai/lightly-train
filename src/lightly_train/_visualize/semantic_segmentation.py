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
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize import utils
from lightly_train.types import MaskSemanticSegmentationBatch


def plot_semantic_segmentation_labels(
    batch: MaskSemanticSegmentationBatch,
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with ground truth semantic segmentation masks.

    Args:
        batch: Semantic segmentation batch with images and masks. Mask pixel values
            are internal contiguous class indices, not original class ids. Masks
            may also contain the raw class_ignore_index value (e.g. -100), which
            is not a contiguous internal class id.
        class_names: A dict mapping internal class IDs to class names. May
            also contain class_ignore_index mapped to "ignored" when masks
            include ignored pixels.
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
    masks = batch["mask"]
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    gt_masks = [m.cpu() for m in masks] if isinstance(masks, list) else masks.cpu()
    n = min(max_images, len(gt_images))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if image_normalize is not None:
            image_tensor = utils._denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )
        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        mask = gt_masks[i]

        overlay = utils._build_semantic_mask_overlay(
            mask=mask, size=img.size, class_names=class_names
        )
        blended = Image.blend(img, overlay, alpha=alpha)
        blended = utils._draw_mask_contours(image=blended, mask=mask)

        labels, colors = utils._legend_entries_for_mask(
            mask=mask, class_names=class_names
        )
        blended = utils._draw_class_legend(image=blended, labels=labels, colors=colors)

        pil_images.append(blended)

    return utils._render_grid(pil_images)


def plot_semantic_segmentation_predictions(
    batch: MaskSemanticSegmentationBatch,
    logits: list[Tensor],
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with predicted semantic segmentation masks.

    Args:
        batch: Semantic segmentation batch with images.
        logits: List of per-image logit tensors of shape (C, H, W).
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
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    n = min(max_images, len(gt_images))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if image_normalize is not None:
            image_tensor = utils._denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )

        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        logits_i = logits[i].cpu()
        pred_mask = torch.argmax(logits_i, dim=0)

        overlay = utils._build_semantic_mask_overlay(
            mask=pred_mask, size=img.size, class_names=class_names
        )
        blended = Image.blend(img, overlay, alpha=alpha)
        blended = utils._draw_mask_contours(image=blended, mask=pred_mask)

        labels, colors = utils._legend_entries_for_mask(
            mask=pred_mask, class_names=class_names
        )
        blended = utils._draw_class_legend(image=blended, labels=labels, colors=colors)

        pil_images.append(blended)

    return utils._render_grid(pil_images)
