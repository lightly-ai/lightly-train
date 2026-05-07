#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import io
from collections.abc import Sequence

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
from lightly_train.types import MaskPanopticSegmentationBatch


def plot_panoptic_segmentation_labels(
    batch: MaskPanopticSegmentationBatch,
    included_classes: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with ground truth panoptic segmentation masks.

    The class label channel of each mask is colorized using ``included_classes``,
    and contours are drawn along segment boundaries so that distinct instances
    of the same class remain visually separated.

    Args:
        batch: Panoptic segmentation batch with images and (H, W, 2) masks where
            channel 0 holds internal class labels and channel 1 holds segment
            ids (segment id -1 indicates unassigned pixels).
        included_classes: A dict mapping internal class IDs to class names. May
            also contain the internal ignore class id mapped to "ignored" when
            masks include ignored pixels.
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
    masks = batch["masks"]
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    gt_masks = [m.cpu() for m in masks] if isinstance(masks, list) else masks.cpu()
    n = min(max_images, len(gt_images))

    pil_images = [
        _render_panoptic_image(
            image=gt_images[i],
            mask=gt_masks[i],
            included_classes=included_classes,
            image_normalize=image_normalize,
            alpha=alpha,
        )
        for i in range(n)
    ]
    return _render_grid(pil_images)


def plot_panoptic_segmentation_predictions(
    batch: MaskPanopticSegmentationBatch,
    pred_masks: Sequence[Tensor],
    included_classes: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with predicted panoptic segmentation masks.

    Args:
        batch: Panoptic segmentation batch with images.
        pred_masks: Per-image predicted (H, W, 2) masks where channel 0 holds
            internal class labels and channel 1 holds segment ids. Each mask
            must have the same spatial size as its corresponding image in
            ``batch["image"]``.
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
    gt_images = (
        [img.cpu() for img in images] if isinstance(images, list) else images.cpu()
    )
    n = min(max_images, len(gt_images), len(pred_masks))

    pil_images = [
        _render_panoptic_image(
            image=gt_images[i],
            mask=pred_masks[i].cpu(),
            included_classes=included_classes,
            image_normalize=image_normalize,
            alpha=alpha,
        )
        for i in range(n)
    ]
    return _render_grid(pil_images)


def _render_panoptic_image(
    image: Tensor,
    mask: Tensor,
    included_classes: dict[int, str],
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a single image with its panoptic mask overlay, contours, and legend."""
    image_tensor = image.clone()
    if image_normalize is not None:
        image_tensor = _denormalize_image(
            image=image_tensor,
            mean=image_normalize["mean"],
            std=image_normalize["std"],
        )
    img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
    # (H, W, 2): channel 0 is the class label, channel 1 is the segment id.
    label_mask = mask[..., 0]
    segment_mask = mask[..., 1]

    overlay = _build_mask_overlay(
        label_mask=label_mask,
        segment_mask=segment_mask,
        size=img.size,
        class_names=included_classes,
    )
    blended = Image.blend(img, overlay, alpha=alpha)
    # Use segment id boundaries so different instances of the same class
    # remain visually separated.
    blended = _draw_mask_contours(image=blended, mask=segment_mask)

    labels, colors = _legend_entries_for_mask(
        mask=label_mask, class_names=included_classes
    )
    return _draw_class_legend(image=blended, labels=labels, colors=colors)


def _build_mask_overlay(
    label_mask: Tensor,
    segment_mask: Tensor,
    size: tuple[int, int],
    class_names: dict[int, str],
) -> PILImage:
    """Build an RGB overlay image where each segment is colored.

    Different instances of the same class are given visually-close but distinct
    colors by passing ``class_id + small_offset`` to ``_get_class_color``. The
    first instance of each class uses the base class color so that the legend
    color matches.

    Only class ids that appear as keys of ``class_names`` are colored; pixels
    with any other id are left black. The ignore index is colored when callers
    include it as a key in ``class_names`` (e.g. mapped to ``"ignored"``) and
    left black otherwise.

    Args:
        label_mask: Tensor of shape (H, W) with internal contiguous class
            indices.
        segment_mask: Tensor of shape (H, W) with per-pixel segment ids.
            Different segment ids within the same class are treated as
            different instances and rendered with different colors.
        size: Target (width, height) of the overlay.
        class_names: Mapping from class id to class name. Only ids that appear
            as keys are colored; other ids are left black.

    Returns:
        RGB PIL image of the requested size.
    """
    h, w = label_mask.shape[-2:]
    overlay = torch.zeros((3, h, w), dtype=torch.uint8)
    instance_hue_step = 0.01
    for class_id in torch.unique(label_mask).tolist():
        class_id = int(class_id)
        if class_id not in class_names:
            continue
        class_pixels = label_mask == class_id
        segment_ids = sorted(
            int(s) for s in torch.unique(segment_mask[class_pixels]).tolist()
        )
        for instance_idx, seg_id in enumerate(segment_ids):
            color = _get_class_color(class_id + instance_hue_step * instance_idx)
            seg_pixels = class_pixels & (segment_mask == seg_id)
            for c in range(3):
                overlay[c][seg_pixels] = color[c]

    overlay_img: PILImage = Image.fromarray(overlay.permute(1, 2, 0).numpy()).convert(
        "RGB"
    )
    if overlay_img.size != size:
        overlay_img = overlay_img.resize(size, resample=Image.Resampling.NEAREST)
    return overlay_img


def _draw_mask_contours(
    image: PILImage,
    mask: Tensor,
) -> PILImage:
    """Overlay thin black contours along boundaries of ``mask`` onto ``image``.

    Boundary pixels are pixels whose value differs from any 4-connected
    neighbor. The contours are drawn after blending so that they remain solid
    black and are not faded by the overlay alpha.

    Args:
        image: RGB PIL image to draw contours on.
        mask: Tensor of shape (H, W) used to detect boundaries.

    Returns:
        A new RGB PIL image with boundaries marked in black.
    """
    h, w = mask.shape[-2:]
    boundary = torch.zeros((h, w), dtype=torch.bool)
    diff_v = mask[:-1, :] != mask[1:, :]
    boundary[:-1, :] |= diff_v
    boundary[1:, :] |= diff_v
    diff_h = mask[:, :-1] != mask[:, 1:]
    boundary[:, :-1] |= diff_h
    boundary[:, 1:] |= diff_h

    boundary_img = Image.fromarray((boundary.to(torch.uint8) * 255).numpy())
    if boundary_img.size != image.size:
        boundary_img = boundary_img.resize(
            image.size, resample=Image.Resampling.NEAREST
        )

    result = image.copy()
    black = Image.new("RGB", image.size, (0, 0, 0))
    result.paste(black, mask=boundary_img)
    return result


def _legend_entries_for_mask(
    mask: Tensor,
    class_names: dict[int, str],
) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Build legend labels and colors for the unique classes present in ``mask``.

    Entries are sorted by class id and skip classes that are not in
    ``class_names``.
    """
    labels: list[str] = []
    colors: list[tuple[int, int, int]] = []
    for class_id in sorted(int(c) for c in torch.unique(mask).tolist()):
        class_name = class_names.get(class_id)
        if class_name is None:
            continue
        labels.append(str(class_name))
        colors.append(_get_class_color(class_id))
    return labels, colors


def _draw_class_legend(
    image: PILImage,
    labels: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
) -> PILImage:
    """Composite a colored-patch legend onto the upper-left of ``image``.

    The legend is rendered on a transparent canvas via matplotlib's headless
    Agg backend and composited onto the image, so pixels outside the legend
    area are preserved unchanged. Returns the image unchanged when ``labels``
    is empty.

    Args:
        image: Base PIL image to render on.
        labels: Legend lines to display, in order.
        colors: Per-label RGB colors in [0, 255]; must have the same length as
            ``labels``.

    Returns:
        A new RGB PIL image with the legend baked in (or the input image when
        ``labels`` is empty).
    """
    if not labels:
        return image
    if len(colors) != len(labels):
        raise ValueError(
            f"colors and labels must have the same length, got "
            f"{len(colors)} and {len(labels)}."
        )

    handles = [
        mpatches.Patch(
            color=(r / 255, g / 255, b / 255),
            label=label,
        )
        for label, (r, g, b) in zip(labels, colors)
    ]

    img_width, img_height = image.size
    dpi = 100
    fig = Figure(figsize=(img_width / dpi, img_height / dpi), dpi=dpi)
    fig.patch.set_alpha(0)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.7,
        fontsize=10,
        borderpad=0.4,
        labelspacing=0.3,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
    buf.seek(0)
    overlay = Image.open(buf).convert("RGBA")
    if overlay.size != (img_width, img_height):
        overlay = overlay.resize(
            (img_width, img_height), resample=Image.Resampling.BILINEAR
        )

    base = image.convert("RGBA")
    base.alpha_composite(overlay)
    return base.convert("RGB")
