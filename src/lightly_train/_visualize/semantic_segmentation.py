#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import io
import matplotlib.patches as mpatches
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize.utils import (
    _denormalize_image,
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
    alpha: float = 0.6,
) -> PILImage:
    """Render a grid of images annotated with ground truth semantic segmentation masks.

    Args:
        batch: Semantic segmentation batch with images and masks. Mask pixel values
            are internal contiguous class indices, not original class ids.
        class_names: Mapping from internal class id to class name.
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

        overlay = _build_mask_overlay(mask=mask, size=img.size)
        blended = Image.blend(img, overlay, alpha=alpha)
        blended = _draw_mask_contours(image=blended, mask=mask)

        blended = _draw_class_legend(
            image=blended, mask=mask, class_names=class_names
        )

        pil_images.append(blended)

    return _render_grid(pil_images)


def plot_semantic_segmentation_predictions(
    batch: MaskSemanticSegmentationBatch,
    predictions: list[Tensor],
    class_names: dict[int, str],
    max_images: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    alpha: float = 0.6,
) -> PILImage:
    """Render a grid of images annotated with predicted semantic segmentation masks.

    Args:
        batch: Semantic segmentation batch with images.
        predictions: List of per-image logit tensors of shape (C, H, W).
        class_names: Mapping from internal class id to class name.
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
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    n = min(max_images, len(gt_images))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if mean is not None and std is not None:
            image_tensor = _denormalize_image(image=image_tensor, mean=mean, std=std)

        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        pred_logits_i = predictions[i].cpu()
        pred_mask = torch.argmax(pred_logits_i, dim=0)

        overlay = _build_mask_overlay(mask=pred_mask, size=img.size)
        blended = Image.blend(img, overlay, alpha=alpha)
        blended = _draw_mask_contours(image=blended, mask=pred_mask)

        blended = _draw_class_legend(
            image=blended, mask=pred_mask, class_names=class_names
        )

        pil_images.append(blended)

    return _render_grid(pil_images)


def _build_mask_overlay(
    mask: Tensor,
    size: tuple[int, int],
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


def _draw_mask_contours(
    image: PILImage,
    mask: Tensor,
) -> PILImage:
    """Overlay thin black contours along class boundaries of ``mask`` onto ``image``.

    The contours are drawn after blending so that they remain solid black and
    are not faded by the overlay alpha.

    Args:
        image: RGB PIL image to draw contours on.
        mask: Tensor of shape (H, W) with internal contiguous class indices.

    Returns:
        A new RGB PIL image with class boundaries marked in black.
    """
    h, w = mask.shape[-2:]
    boundary = torch.zeros((h, w), dtype=torch.bool)
    diff_v = mask[:-1, :] != mask[1:, :]
    boundary[:-1, :] |= diff_v
    boundary[1:, :] |= diff_v
    diff_h = mask[:, :-1] != mask[:, 1:]
    boundary[:, :-1] |= diff_h
    boundary[:, 1:] |= diff_h

    boundary_img = torchvision_functional.to_pil_image(
        boundary.to(torch.uint8) * 255
    )
    if boundary_img.size != image.size:
        boundary_img = boundary_img.resize(
            image.size, resample=Image.Resampling.NEAREST
        )

    result = image.copy()
    black = Image.new("RGB", image.size, (0, 0, 0))
    result.paste(black, mask=boundary_img)
    return result


def _draw_class_legend(
    image: PILImage,
    mask: Tensor,
    class_names: dict[int, str],
) -> PILImage:
    """Render the image with a matplotlib legend of class colors and names.

    Builds one legend entry per unique class id present in ``mask``, sorted by
    class id, with the patch color matching the mask overlay. Returns a new
    PIL image with the legend baked in.
    """
    handles = []
    for class_id in sorted(int(c) for c in torch.unique(mask).tolist()):
        class_name = class_names.get(class_id)
        if class_name is None:
            continue
        r, g, b = _get_class_color(class_id)
        handles.append(
            mpatches.Patch(
                color=(r / 255, g / 255, b / 255),
                label=str(class_name),
            )
        )

    if not handles:
        return image

    img_width, img_height = image.size
    dpi = 100
    fig = Figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_axis_off()
    ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.7,
        fontsize=10,
        borderpad=0.4,
        labelspacing=0.3,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    buf.seek(0)
    rendered = Image.open(buf).convert("RGB")
    if rendered.size != (img_width, img_height):
        rendered = rendered.resize(
            (img_width, img_height), resample=Image.Resampling.BILINEAR
        )
    return rendered