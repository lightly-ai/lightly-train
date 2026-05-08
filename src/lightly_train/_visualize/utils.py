#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import colorsys
import io
import math
from collections.abc import Sequence

import matplotlib.patches as mpatches
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageFont
from PIL.Image import Image as PILImage
from PIL.ImageDraw import ImageDraw as PILDraw
from torch import Tensor

try:
    _DEFAULT_FONT = ImageFont.load_default(size=20)
except TypeError:
    _DEFAULT_FONT = ImageFont.load_default()


def _draw_class_legend(
    image: PILImage,
    labels: Sequence[str],
    colors: Sequence[tuple[int, int, int]] | None,
) -> PILImage:
    """Composite a legend onto the upper-left of ``image``.

    The legend is rendered on a transparent canvas via matplotlib's headless
    Agg backend and composited onto the image, so pixels outside the legend
    area are preserved unchanged. Returns the image unchanged when ``labels``
    is empty.

    Args:
        image: Base PIL image to render on.
        labels: Legend lines to display, in order.
        colors: Optional per-label RGB colors in [0, 255]. When provided, each
            entry is rendered as a colored patch next to its label; must have
            the same length as ``labels``. When None, labels are rendered as
            text only.

    Returns:
        A new RGB PIL image with the legend baked in (or the input image when
        ``labels`` is empty).
    """
    if not labels:
        return image

    if colors is None:
        handles = [
            mpatches.Patch(facecolor="none", edgecolor="none", label=label)
            for label in labels
        ]
        legend_kwargs: dict[str, object] = dict(
            borderaxespad=0,
            handlelength=0,
            handletextpad=0,
        )
    else:
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
        legend_kwargs = {}

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
        **legend_kwargs,
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


def _draw_bbox_label(
    draw: PILDraw,
    x1: float,
    y1: float,
    text: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a highlighted label rectangle near a bounding box.

    Draws above the box when there is enough space; otherwise draws below.
    """
    padding = 4

    bbox = draw.textbbox((0, 0), text, font=_DEFAULT_FONT)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x1 = float(x1)
    y1 = float(y1)

    label_width = text_width + 2 * padding
    label_height = text_height + 2 * padding

    if y1 >= label_height:
        rect_top = y1 - label_height
        rect_bottom = y1
    else:
        rect_top = y1
        rect_bottom = y1 + label_height

    rect_left = x1
    rect_right = x1 + label_width

    x0, x1_rect = sorted((rect_left, rect_right))
    y0, y1_rect = sorted((rect_top, rect_bottom))

    draw.rectangle([x0, y0, x1_rect, y1_rect], fill=color, outline=color)
    draw.text((x0 + padding, y0 + padding), text, fill="white", font=_DEFAULT_FONT)


def _denormalize_image(
    image: Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    """Denormalize an image tensor using mean and std.

    Args:
        image: Tensor of shape (3, H, W) with normalized values.
        mean: Tuple of mean values used for normalization.
        std: Tuple of std values used for normalization.

    Returns:
        Denormalized tensor with values clamped to the [0, 1] range.
    """
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(
        -1, 1, 1
    )
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(
        -1, 1, 1
    )

    denormalized = image * std_tensor + mean_tensor
    denormalized = torch.clamp(denormalized, 0, 1)

    return denormalized


def _get_class_color(class_id: float) -> tuple[int, int, int]:
    """Generate a deterministic RGB color for a class ID.

    The hue is computed as ``(class_id * golden_ratio) % 1.0``. Multiplying by
    the golden ratio spreads consecutive integer class IDs across the hue
    circle so different classes are well-separated, while a small fractional
    offset on ``class_id`` produces only a small hue shift. Callers can
    therefore pass ``class_id + small_offset`` to derive instance-specific
    colors that remain in the same color region as the base class.

    Args:
        class_id: The class ID to generate a color for.

    Returns:
        RGB tuple with values in range [0, 255].
    """
    hue = (
        class_id * 0.618033988749895
    ) % 1.0  # Use the golden ratio for good hue distribution.

    # Use high saturation and value for vibrant, distinct colors.
    saturation = 0.9
    value = 0.95

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _render_grid(pil_images: list[PILImage]) -> PILImage:
    """Arrange PIL images into a square-ish grid.

    Tiles may have different sizes. The cell size is the maximum width and
    height across all tiles, and each tile is centered within its cell.

    Args:
        pil_images: List of PIL images.

    Returns:
        Single PIL image with all inputs tiled into a grid.
    """
    n = len(pil_images)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    w = max(img.size[0] for img in pil_images)
    h = max(img.size[1] for img in pil_images)
    mode = pil_images[0].mode
    grid = Image.new(mode, (n_cols * w, n_rows * h))
    for idx, img in enumerate(pil_images):
        row, col = divmod(idx, n_cols)
        img_w, img_h = img.size
        x = col * w + (w - img_w) // 2
        y = row * h + (h - img_h) // 2
        grid.paste(img, (x, y))
    return grid


def _build_instance_mask_overlay(
    masks: Tensor,
    labels: Tensor,
    size: tuple[int, int],
    class_names: dict[int, str],
) -> PILImage:
    """Build an RGB overlay image colored by class id from per-instance binary masks.

    Only ids that appear as keys of ``class_names`` are colored; pixels with
    any other id are left black.

    Args:
        masks: Boolean tensor of shape (n_instances, H, W).
        labels: Tensor of shape (n_instances,) with internal class ids.
        size: Target (width, height) of the overlay.
        class_names: Mapping from class id to class name. Only ids that appear
            as keys are colored; other ids are left black.

    Returns:
        RGB PIL image of the requested size.
    """
    h, w = masks.shape[-2:]
    overlay = torch.zeros((3, h, w), dtype=torch.uint8)
    for idx in range(masks.shape[0]):
        class_id = int(labels[idx])
        if class_id not in class_names:
            continue
        color = _get_class_color(class_id)
        for c in range(3):
            overlay[c][masks[idx]] = color[c]

    return _overlay_to_pil(overlay=overlay, size=size)


def _build_panoptic_mask_overlay(
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

    return _overlay_to_pil(overlay=overlay, size=size)


def _build_semantic_mask_overlay(
    mask: Tensor,
    size: tuple[int, int],
    class_names: dict[int, str],
) -> PILImage:
    """Build an RGB overlay image colored by class id from a 2D class mask.

    Only ids that appear as keys of ``class_names`` are colored; pixels with
    any other id are left black.

    Args:
        mask: Tensor of shape (H, W) with internal class ids per pixel.
        size: Target (width, height) of the overlay.
        class_names: Mapping from class id to class name. Only ids that appear
            as keys are colored; other ids are left black.

    Returns:
        RGB PIL image of the requested size.
    """
    h, w = mask.shape[-2:]
    overlay = torch.zeros((3, h, w), dtype=torch.uint8)
    for class_id in torch.unique(mask).tolist():
        class_id = int(class_id)
        if class_id not in class_names:
            continue
        color = _get_class_color(class_id)
        pixels = mask == class_id
        for c in range(3):
            overlay[c][pixels] = color[c]

    return _overlay_to_pil(overlay=overlay, size=size)


def _overlay_to_pil(overlay: Tensor, size: tuple[int, int]) -> PILImage:
    """Convert a (3, H, W) uint8 overlay tensor to a resized RGB PIL image."""
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
    ``class_names`` (e.g. ignore_index).
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


def _bboxes_from_masks(masks: Tensor) -> tuple[Tensor, Tensor]:
    """Derive xyxy bounding boxes from per-instance binary masks.

    Empty masks (no foreground pixels) are skipped: their entries do not appear
    in the returned boxes and the corresponding entry in the keep tensor is
    False. Callers should use the keep tensor to filter parallel arrays such as
    labels and scores.

    Args:
        masks: Boolean tensor of shape (n_instances, H, W).

    Returns:
        A tuple (boxes, keep) where boxes has shape (n_kept, 4) in xyxy pixel
        coordinates and keep is a boolean tensor of shape (n_instances,).
    """
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), dtype=torch.float32)
    keep = torch.zeros((n,), dtype=torch.bool)
    for i in range(n):
        ys, xs = torch.where(masks[i])
        if ys.numel() == 0:
            continue
        boxes[i, 0] = float(xs.min())
        boxes[i, 1] = float(ys.min())
        boxes[i, 2] = float(xs.max())
        boxes[i, 3] = float(ys.max())
        keep[i] = True
    return boxes[keep], keep


def _draw_labeled_boxes(
    image: PILImage,
    bboxes_xyxy: Tensor,
    labels: Tensor,
    scores: Tensor | None,
    class_names: dict[int, str],
) -> None:
    """Draw a colored bounding box and class label per box, in place.

    Args:
        image: RGB PIL image to draw onto.
        bboxes_xyxy: Tensor of shape (n_boxes, 4) in xyxy pixel coordinates.
        labels: Tensor of shape (n_boxes,) with internal class ids.
        scores: Optional tensor of shape (n_boxes,) with per-box scores. When
            provided, scores are appended to each label.
        class_names: A dict mapping internal class IDs to class names.
    """
    if bboxes_xyxy.shape[0] == 0:
        return
    draw = PILDraw(image)
    for i in range(bboxes_xyxy.shape[0]):
        x1, y1, x2, y2 = bboxes_xyxy[i].tolist()
        class_id = int(labels[i])
        class_name = class_names.get(class_id, f"Class {class_id}")
        color = _get_class_color(class_id)
        text = class_name if scores is None else f"{class_name} {float(scores[i]):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        _draw_bbox_label(draw=draw, x1=x1, y1=y1, text=text, color=color)
