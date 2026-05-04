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
    _denormalize_image,
    _draw_bbox_label,
    _get_class_color,
    _render_grid,
)
from lightly_train.types import MaskSemanticSegmentationBatch


def plot_semantic_segmentation_labels(
    batch: MaskSemanticSegmentationBatch,
    class_names: dict[int, str],
    max_images: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    alpha: float = 0.5,
) -> PILImage:
    """Render a grid of images annotated with ground truth semantic segmentation masks.

    Args:
        batch: Semantic segmentation batch with images and masks. Mask pixel values
            are internal contiguous class indices, not original class ids.
        included_classes: Mapping from original class id to class name.
        class_id_to_internal_class_id: Mapping from original class id to the internal
            contiguous class index used in the batch masks.
        max_images: Maximum number of images to include in the grid.
        mean: Per-channel mean used for image normalization (for denormalization).
        std: Per-channel std used for image normalization (for denormalization).
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
        if mean is not None and std is not None:
            image_tensor = _denormalize_image(image=image_tensor, mean=mean, std=std)

        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        mask = gt_masks[i]

        overlay = _build_mask_overlay(
            mask=mask, size=img.size
        )
        blended = Image.blend(img, overlay, alpha=alpha)

        img_width, img_height = blended.size
        scale_x = img_width / mask.shape[-1]
        scale_y = img_height / mask.shape[-2]
        draw = PILDraw(blended)
        for class_id in sorted(int(c) for c in torch.unique(mask).tolist()):
            class_name = class_names.get(class_id)
            if class_name is None:
                continue
            class_pixels = (mask == class_id)
            ys, xs = torch.where(class_pixels)
            if xs.numel() == 0:
                continue
            x1 = float(xs.min().item()) * scale_x
            y1 = float(ys.min().item()) * scale_y
            x2 = float(xs.max().item() + 1) * scale_x
            y2 = float(ys.max().item() + 1) * scale_y
            color = _get_class_color(class_id)
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


def _build_mask_overlay(
    mask: Tensor,
    size: tuple[int, int]
) -> PILImage:
    """Build an RGB overlay image where each pixel is colored by its class id.

    Pixels whose class id is not present in the mapping (e.g.
    ignore_index) are left black.

    Args:
        mask: Tensor of shape (H, W) with internal contiguous class indices.
        size: Target (width, height) of the overlay.

    Returns:
        RGB PIL image of the requested size.
    """
    h, w = mask.shape[-2:]
    overlay = torch.zeros((3, h, w), dtype=torch.uint8)
    for class_id in torch.unique(mask).tolist():
        class_id = int(class_id)
        color = _get_class_color(class_id)
        class_pixels = mask == class_id
        for c in range(3):
            overlay[c][class_pixels] = color[c]

    overlay_img: PILImage = torchvision_functional.to_pil_image(overlay).convert("RGB")
    if overlay_img.size != size:
        overlay_img = overlay_img.resize(size, resample=Image.Resampling.NEAREST)
    return overlay_img
