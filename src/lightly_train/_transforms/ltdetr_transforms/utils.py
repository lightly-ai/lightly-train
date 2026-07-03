#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from albumentations import (
    BasicTransform,
    HorizontalFlip,
    RandomRotate90,
    Resize,
    Rotate,
    ToFloat,
    VerticalFlip,
)
from lightning_utilities.core.imports import RequirementCache

from lightly_train._configs.validate import no_auto
from lightly_train._task_models.object_detection_components.ltdetr_geometry import (
    ltdetr_image_size_divisor,
)
from lightly_train._task_models.object_detection_components.ltdetr_schedule import (
    resolve_ltdetr_step_schedule,
)
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.random_iou_crop import RandomIoUCrop
from lightly_train._transforms.random_photometric_distort import (
    RandomPhotometricDistort,
)
from lightly_train._transforms.random_zoom_out import RandomZoomOut
from lightly_train._transforms.transform import ActivationPolicyArgs
from lightly_train.types import NDArrayBBoxes, NDArrayClasses

if TYPE_CHECKING:
    from lightly_train._transforms.ltdetr_transforms.base import LTDETRTransformArgs

logger = logging.getLogger(__name__)

ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_5 = RequirementCache("albumentations>=1.4.5")
ALBUMENTATIONS_VERSION_GREATER_EQUAL_2_0_1 = RequirementCache("albumentations>=2.0.1")


def resolve_image_size_for_patch_size(
    model_init_args: dict[str, Any],
    *,
    default_image_size: tuple[int, int],
    patch_size: int | None,
) -> tuple[int, int]:
    provided_image_size = model_init_args.get("image_size")
    if provided_image_size is not None:
        image_size = (
            int(provided_image_size[0]),
            int(provided_image_size[1]),
        )
        if patch_size is not None:
            divisor = ltdetr_image_size_divisor(patch_size)
            if any(size % divisor != 0 for size in image_size):
                raise ValueError(
                    "When providing an image size in model_init_args, it must be divisible by 2 * the patch size."
                )
        return image_size

    if patch_size is None:
        return default_image_size

    divisor = ltdetr_image_size_divisor(patch_size)
    return (
        math.ceil(default_image_size[0] / divisor) * divisor,
        math.ceil(default_image_size[1] / divisor) * divisor,
    )


def resolve_ltdetr_step_schedule_for_augmentation(
    args: LTDETRTransformArgs,
    total_steps: int,
    train_num_batches: int,
    gradient_accumulation_steps: int,
) -> None:
    """Resolve LT-DETR augmentation step windows from ``"auto"``.

    - ``photometric_distort``, ``random_zoom_out``, ``random_iou_crop``, and
      ``copyblend`` use [``step_start_resolved``, ``step_stop_resolved``)
    - ``mixup`` and ``mosaic`` use [``step_start_resolved``, ``step_flat_resolved``)
    - ``scale_jitter`` only resolves ``step_stop_resolved``

    If an augmentation's final integer window is empty, it is disabled instead
    of clamped to a minimum length. In practice this means:
    - ``step_stop <= step_start`` disables the corresponding augmentation field
    - ``scale_jitter`` is disabled when its resolved auto ``step_stop <= 0``
    """
    step_schedule = resolve_ltdetr_step_schedule(
        total_steps=total_steps,
        train_num_batches=train_num_batches,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    _resolve_aug_fields(
        args=args,
        field_names=(
            "photometric_distort",
            "random_zoom_out",
            "random_iou_crop",
            "copyblend",
        ),
        step_start_resolved=step_schedule.step_start,
        step_stop_resolved=step_schedule.step_stop,
    )
    _resolve_aug_fields(
        args=args,
        field_names=("mixup", "mosaic"),
        step_start_resolved=step_schedule.step_start,
        step_stop_resolved=step_schedule.step_flat,
    )

    scale_jitter = args.scale_jitter
    if scale_jitter is not None and scale_jitter.step_stop == "auto":
        if step_schedule.step_stop <= 0:
            logger.warning(
                "Auto-derived step window for scale_jitter has "
                f"step_stop ({step_schedule.step_stop}) <= 0. "
                "Disabling scale_jitter for this run."
            )
            args.scale_jitter = None
        else:
            scale_jitter.step_stop = step_schedule.step_stop


def _resolve_aug_fields(
    args: LTDETRTransformArgs,
    field_names: tuple[str, ...],
    step_start_resolved: int,
    step_stop_resolved: int,
) -> None:
    for field_name in field_names:
        aug = getattr(args, field_name, None)
        if aug is None:
            continue

        step_start = step_start_resolved if aug.step_start == "auto" else aug.step_start
        step_stop = step_stop_resolved if aug.step_stop == "auto" else aug.step_stop

        if (
            isinstance(step_start, int)
            and isinstance(step_stop, int)
            and step_stop <= step_start
        ):
            logger.warning(
                f"Auto-derived step window for {field_name} has "
                f"step_stop ({step_stop}) <= step_start ({step_start}). "
                f"Disabling {field_name} for this run."
            )
            setattr(args, field_name, None)
            continue

        if aug.step_start == "auto":
            aug.step_start = step_start
        if aug.step_stop == "auto":
            aug.step_stop = step_stop


def is_step_start_or_stop_configured(
    activation_policy: ActivationPolicyArgs | None,
) -> bool:
    if activation_policy is None:
        return False
    return activation_policy.step_start != 0 or activation_policy.step_stop is not None


def build_ltdetr_sample_transform_parts(
    transform_args: LTDETRTransformArgs,
) -> dict[str, Any]:
    parts: dict[str, Any] = {
        "channel_drop": None,
        "photometric_distort": None,
        "random_zoom_out": None,
        "random_iou_crop": None,
        "horizontal_flip": None,
        "vertical_flip": None,
        "random_rotate_90": None,
        "random_rotate": None,
        "resize": None,
        "to_float": ToFloat(max_value=255.0),
        "normalize": None,
    }

    if transform_args.channel_drop is not None:
        parts["channel_drop"] = ChannelDrop(
            num_channels_keep=transform_args.channel_drop.num_channels_keep,
            weight_drop=transform_args.channel_drop.weight_drop,
        )
    if transform_args.photometric_distort is not None:
        parts["photometric_distort"] = RandomPhotometricDistort(
            brightness=transform_args.photometric_distort.brightness,
            contrast=transform_args.photometric_distort.contrast,
            saturation=transform_args.photometric_distort.saturation,
            hue=transform_args.photometric_distort.hue,
            p=transform_args.photometric_distort.prob,
        )
    if transform_args.random_zoom_out is not None:
        parts["random_zoom_out"] = RandomZoomOut(
            fill=transform_args.random_zoom_out.fill,
            side_range=transform_args.random_zoom_out.side_range,
            p=transform_args.random_zoom_out.prob,
        )
    if transform_args.random_iou_crop is not None:
        parts["random_iou_crop"] = RandomIoUCrop(
            min_scale=transform_args.random_iou_crop.min_scale,
            max_scale=transform_args.random_iou_crop.max_scale,
            min_aspect_ratio=transform_args.random_iou_crop.min_aspect_ratio,
            max_aspect_ratio=transform_args.random_iou_crop.max_aspect_ratio,
            sampler_options=transform_args.random_iou_crop.sampler_options,
            crop_trials=transform_args.random_iou_crop.crop_trials,
            iou_trials=transform_args.random_iou_crop.iou_trials,
            p=transform_args.random_iou_crop.prob,
        )
    if transform_args.random_flip is not None:
        if transform_args.random_flip.horizontal_prob > 0.0:
            parts["horizontal_flip"] = HorizontalFlip(
                p=transform_args.random_flip.horizontal_prob
            )
        if transform_args.random_flip.vertical_prob > 0.0:
            parts["vertical_flip"] = VerticalFlip(
                p=transform_args.random_flip.vertical_prob
            )
    if transform_args.random_rotate_90 is not None:
        parts["random_rotate_90"] = RandomRotate90(
            p=transform_args.random_rotate_90.prob
        )
    if transform_args.random_rotate is not None:
        parts["random_rotate"] = Rotate(
            limit=transform_args.random_rotate.degrees,
            interpolation=transform_args.random_rotate.interpolation,
            p=transform_args.random_rotate.prob,
        )
    if transform_args.resize is not None:
        parts["resize"] = Resize(
            height=no_auto(transform_args.resize.height),
            width=no_auto(transform_args.resize.width),
        )
    if transform_args.normalize is not None:
        parts["normalize"] = Normalize(
            mean=no_auto(transform_args.normalize).mean,
            std=no_auto(transform_args.normalize).std,
            max_pixel_value=1.0,
        )
    return parts


def ordered_ltdetr_sample_transforms(
    parts: dict[str, Any],
    *,
    photometric_distort_active: bool,
    random_zoom_out_active: bool,
    random_iou_crop_active: bool,
) -> list[BasicTransform]:
    transforms: list[BasicTransform] = []
    for key, active in (
        ("channel_drop", True),
        ("photometric_distort", photometric_distort_active),
        ("random_zoom_out", random_zoom_out_active),
        ("random_iou_crop", random_iou_crop_active),
        ("horizontal_flip", True),
        ("vertical_flip", True),
        ("random_rotate_90", True),
        ("random_rotate", True),
        ("resize", True),
        ("to_float", True),
        ("normalize", True),
    ):
        transform = parts[key]
        if transform is not None and active:
            transforms.append(transform)
    return transforms


def normalize_bboxes_and_labels(
    bboxes: Any, class_labels: Any
) -> tuple[NDArrayBBoxes, NDArrayClasses]:
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    if isinstance(class_labels, list):
        class_labels = np.array(class_labels)
    return bboxes, class_labels


def filter_degenerate_yolo_boxes(
    bboxes: NDArrayBBoxes,
    class_labels: NDArrayClasses,
    indices: np.ndarray | None = None,
) -> tuple[NDArrayBBoxes, NDArrayClasses, np.ndarray | None]:
    if len(bboxes) == 0:
        return bboxes, class_labels, indices
    valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
    if indices is None:
        return bboxes[valid], class_labels[valid], None
    return bboxes[valid], class_labels[valid], indices[valid]
