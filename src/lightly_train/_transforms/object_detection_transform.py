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
from typing import Any, Literal

import numpy as np
import torch
from albumentations import (
    BboxParams,
    Compose,
    HorizontalFlip,
    RandomRotate90,
    Resize,
    Rotate,
    ToFloat,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2
from pydantic import ConfigDict
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.batch_transform import BatchReplayCompose, BatchTransform
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.copyblend import CopyBlend
from lightly_train._transforms.mixup import MixUp
from lightly_train._transforms.mosaic import MosaicTransform
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.random_iou_crop import RandomIoUCrop
from lightly_train._transforms.random_photometric_distort import (
    RandomPhotometricDistort,
)
from lightly_train._transforms.random_zoom_out import RandomZoomOut
from lightly_train._transforms.scale_jitter import ScaleJitter
from lightly_train._transforms.task_transform import (
    TaskCollateFunction,
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train._transforms.transform import (
    ActivationPolicyArgs,
    ChannelDropArgs,
    CopyBlendArgs,
    MixUpArgs,
    MosaicArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
)
from lightly_train.types import (
    ImageSizeTuple,
    NDArrayBBoxes,
    NDArrayClasses,
    NDArrayImage,
    ObjectDetectionBatch,
    ObjectDetectionDatasetItem,
)


class ObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NotRequired[NDArrayBBoxes]
    class_labels: NotRequired[NDArrayClasses]


class ObjectDetectionTransformOutput(TaskTransformOutput):
    image: NDArrayImage
    bboxes: NotRequired[NDArrayBBoxes]
    class_labels: NotRequired[NDArrayClasses]


class ObjectDetectionTransformArgs(TaskTransformArgs):
    channel_drop: ChannelDropArgs | None
    num_channels: int | Literal["auto"]
    photometric_distort: RandomPhotometricDistortArgs | None
    random_zoom_out: RandomZoomOutArgs | None
    random_iou_crop: RandomIoUCropArgs | None
    random_flip: RandomFlipArgs | None
    random_rotate_90: RandomRotate90Args | None
    random_rotate: RandomRotationArgs | None
    image_size: ImageSizeTuple | Literal["auto"]
    mixup: MixUpArgs | None = None
    copyblend: CopyBlendArgs | None = None
    mosaic: MosaicArgs | None = None
    scale_jitter: ScaleJitterArgs | None = None
    resize: ResizeArgs | None
    bbox_params: BboxParams | None
    normalize: NormalizeArgs | Literal["auto"] | None

    # Necessary for BboxParams, which are not serializable by pydantic.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        pass

    def resolve_incompatible(self) -> None:
        # TODO: Lionel (09/25): Add checks for incompatible args.
        pass


_logger = logging.getLogger(__name__)

# Matched upstream schedule profile for LTDETR.
_WARMUP_EPOCHS = 4
_SHORT_RUN_TOTAL_EPOCHS = 12
_REFERENCE_TOTAL_EPOCHS = 72
_REFERENCE_NO_AUG_EPOCHS = 12


def resolve_ltdetr_step_schedule(
    args: ObjectDetectionTransformArgs,
    total_steps: int,
    train_num_batches: int,
    gradient_accumulation_steps: int,
) -> None:
    """Resolve ``"auto"`` step_start / step_stop on LTDETR augmentation args.

    The algorithm converts internal epoch semantics into concrete step
    boundaries using the actual dataloader length so that augmentation windows
    adapt to different dataset sizes and batch sizes.

    The calculation works in five stages:

    1. Compute the effective number of optimizer steps per epoch as
       ``train_num_batches / gradient_accumulation_steps``.
    2. Derive the effective training length in epochs as
       ``total_steps / steps_per_epoch``
    3. Scale the canonical LTDETR no-augmentation tail from the matched upstream profile
    (``_REFERENCE_TOTAL_EPOCHS`` / ``_REFERENCE_NO_AUG_EPOCHS``) and scaled
        proportionally to the actual run length::
        no_aug_epochs_resolved = max(
            1,
            min(
                _REFERENCE_NO_AUG_EPOCHS,
                round(total_epochs * _REFERENCE_NO_AUG_EPOCHS / _REFERENCE_TOTAL_EPOCHS),
            ),
        )
    4. Convert the epoch recipe into three epoch boundaries:
        - epoch_stop_resolved = total_epochs - no_aug_epochs_resolved
        - short runs with ``total_epochs <= _SHORT_RUN_TOTAL_EPOCHS`` compress to
            ``epoch_start = min(_WARMUP_EPOCHS, floor(total_epochs / 3))`` and
            ``epoch_flat = min(epoch_stop, 2 * epoch_start)``
        - longer runs keep ``epoch_start = _WARMUP_EPOCHS`` and use
            ``epoch_flat = _WARMUP_EPOCHS + floor(epoch_stop / 2)``
    5. Convert each epoch boundary back to integer steps with
       ``floor(epoch * steps_per_epoch)``.

    The resolved step windows are then assigned as follows:
    - ``photometric_distort``, ``random_zoom_out``, ``random_iou_crop``, and
      ``copyblend`` use ``[step_start, step_stop)``
    - ``mixup`` and ``mosaic`` use ``[step_start, step_flat)``
    - ``scale_jitter`` is stop-only and uses ``[0, step_stop)``

    Collapsed windows are treated as disabling the augmentation instead of causing an error:
    - ``step_stop <= step_start``: disable photometric_distort,
      random_zoom_out, random_iou_crop, and copyblend.
    - ``step_flat <= step_start``: disable mixup and mosaic.
    - ``step_stop <= 0``: disable scale_jitter.
    - An empty window is never clamped to ``step_stop = 1``; the
      augmentation is disabled instead.
    """
    steps_per_epoch = train_num_batches / gradient_accumulation_steps
    total_epochs = total_steps / steps_per_epoch

    # Resolve no-aug tail from matched upstream profile.
    no_aug_epochs_resolved = max(
        1,
        min(
            _REFERENCE_NO_AUG_EPOCHS,
            round(total_epochs * _REFERENCE_NO_AUG_EPOCHS / _REFERENCE_TOTAL_EPOCHS),
        ),
    )
    epoch_stop_resolved = total_epochs - no_aug_epochs_resolved

    # Compute epoch_start and epoch_flat with short-run compression.
    if total_epochs <= _SHORT_RUN_TOTAL_EPOCHS:
        epoch_start_resolved = min(_WARMUP_EPOCHS, math.floor(total_epochs / 3))
        epoch_flat_resolved = min(epoch_stop_resolved, 2 * epoch_start_resolved)
    else:
        epoch_start_resolved = _WARMUP_EPOCHS
        epoch_flat_resolved = _WARMUP_EPOCHS + math.floor(epoch_stop_resolved / 2)

    # Convert epoch boundaries to steps.
    step_start_resolved = math.floor(epoch_start_resolved * steps_per_epoch)
    step_stop_resolved = math.floor(epoch_stop_resolved * steps_per_epoch)
    step_flat_resolved = math.floor(epoch_flat_resolved * steps_per_epoch)

    # Resolve each augmentation field.  We pre-check the final window before
    # assigning so that Pydantic's validate_assignment never sees an invalid
    # step_stop <= step_start pair.
    _resolve_aug_fields(
        args=args,
        field_names=(
            "photometric_distort",
            "random_zoom_out",
            "random_iou_crop",
            "copyblend",
        ),
        step_start_resolved=step_start_resolved,
        step_stop_resolved=step_stop_resolved,
    )
    _resolve_aug_fields(
        args=args,
        field_names=("mixup", "mosaic"),
        step_start_resolved=step_start_resolved,
        step_stop_resolved=step_flat_resolved,
    )

    # Scale jitter: only has step_stop, no step_start.
    if args.scale_jitter is not None and args.scale_jitter.step_stop == "auto":
        if step_stop_resolved <= 0:
            _logger.warning(
                "Auto-derived step window for scale_jitter has "
                f"step_stop ({step_stop_resolved}) <= 0. "
                "Disabling scale_jitter for this run."
            )
            args.scale_jitter = None
        else:
            args.scale_jitter.step_stop = step_stop_resolved


def _resolve_aug_fields(
    args: ObjectDetectionTransformArgs,
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

        # Check if the resolved window is valid before assigning.
        if (
            isinstance(step_start, int)
            and isinstance(step_stop, int)
            and step_stop <= step_start
        ):
            _logger.warning(
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


class ObjectDetectionTransform(TaskTransform):
    transform_args_cls: type[ObjectDetectionTransformArgs] = (
        ObjectDetectionTransformArgs
    )

    def __init__(
        self,
        transform_args: ObjectDetectionTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform_args: ObjectDetectionTransformArgs = transform_args
        self._step = 0
        self.channel_drop: ChannelDrop | None = None
        if transform_args.channel_drop is not None:
            self.channel_drop = ChannelDrop(
                num_channels_keep=transform_args.channel_drop.num_channels_keep,
                weight_drop=transform_args.channel_drop.weight_drop,
            )

        self.photometric_distort: RandomPhotometricDistort | None = None
        if transform_args.photometric_distort is not None:
            self.photometric_distort = RandomPhotometricDistort(
                brightness=transform_args.photometric_distort.brightness,
                contrast=transform_args.photometric_distort.contrast,
                saturation=transform_args.photometric_distort.saturation,
                hue=transform_args.photometric_distort.hue,
                p=transform_args.photometric_distort.prob,
            )

        self.random_zoom_out: RandomZoomOut | None = None
        if transform_args.random_zoom_out is not None:
            self.random_zoom_out = RandomZoomOut(
                fill=transform_args.random_zoom_out.fill,
                side_range=transform_args.random_zoom_out.side_range,
                p=transform_args.random_zoom_out.prob,
            )

        self.random_iou_crop: RandomIoUCrop | None = None
        if transform_args.random_iou_crop is not None:
            self.random_iou_crop = RandomIoUCrop(
                min_scale=transform_args.random_iou_crop.min_scale,
                max_scale=transform_args.random_iou_crop.max_scale,
                min_aspect_ratio=transform_args.random_iou_crop.min_aspect_ratio,
                max_aspect_ratio=transform_args.random_iou_crop.max_aspect_ratio,
                sampler_options=transform_args.random_iou_crop.sampler_options,
                crop_trials=transform_args.random_iou_crop.crop_trials,
                iou_trials=transform_args.random_iou_crop.iou_trials,
                p=transform_args.random_iou_crop.prob,
            )

        self.horizontal_flip: HorizontalFlip | None = None
        self.vertical_flip: VerticalFlip | None = None
        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                self.horizontal_flip = HorizontalFlip(
                    p=transform_args.random_flip.horizontal_prob
                )
            if transform_args.random_flip.vertical_prob > 0.0:
                self.vertical_flip = VerticalFlip(
                    p=transform_args.random_flip.vertical_prob
                )

        self.random_rotate_90: RandomRotate90 | None = None
        if transform_args.random_rotate_90 is not None:
            self.random_rotate_90 = RandomRotate90(
                p=transform_args.random_rotate_90.prob
            )

        self.random_rotate: Rotate | None = None
        if transform_args.random_rotate is not None:
            self.random_rotate = Rotate(
                limit=transform_args.random_rotate.degrees,
                interpolation=transform_args.random_rotate.interpolation,
                p=transform_args.random_rotate.prob,
            )

        self.resize: Resize | None = None
        if transform_args.resize is not None:
            self.resize = Resize(
                height=no_auto(transform_args.resize.height),
                width=no_auto(transform_args.resize.width),
            )

        self.to_float = ToFloat(max_value=255.0)

        self.normalize: Normalize | None = None
        if transform_args.normalize is not None:
            self.normalize = Normalize(
                mean=no_auto(transform_args.normalize).mean,
                std=no_auto(transform_args.normalize).std,
                max_pixel_value=1.0,  # Already scaled.
            )

        self.mosaic: MosaicTransform | None = None
        if transform_args.mosaic is not None:
            self.mosaic = MosaicTransform(
                output_size=transform_args.mosaic.output_size,
                max_size=transform_args.mosaic.max_size,
                rotation_range=transform_args.mosaic.rotation_range,
                translation_range=transform_args.mosaic.translation_range,
                scaling_range=transform_args.mosaic.scaling_range,
                fill_value=int(transform_args.mosaic.fill_value),
                max_cached_images=transform_args.mosaic.max_cached_images,
                random_pop=transform_args.mosaic.random_pop,
            )

        # Rebuilding an albumentations Compose for every sample is unnecessary
        # because these step-aware sample transforms only change at a few step
        # boundaries. Cache the effective pipelines and reuse them per worker.
        self._compose_cache: dict[tuple[bool, bool, bool], Compose] = {}
        self._current_transform_active_status = (
            self._get_transform_active_status_at_step(self._step)
        )

    def _is_photometric_distort_active_at_step(self, step: int) -> bool:
        return (
            self.photometric_distort is not None
            and self.transform_args.photometric_distort is not None
            and self.transform_args.photometric_distort.prob > 0.0
            and self.transform_args.photometric_distort.is_active(step)
        )

    def _is_random_zoom_out_active_at_step(self, step: int) -> bool:
        return (
            self.random_zoom_out is not None
            and self.transform_args.random_zoom_out is not None
            and self.transform_args.random_zoom_out.prob > 0.0
            and self.transform_args.random_zoom_out.is_active(step)
        )

    def _is_random_iou_crop_active_at_step(self, step: int) -> bool:
        return (
            self.random_iou_crop is not None
            and self.transform_args.random_iou_crop is not None
            and self.transform_args.random_iou_crop.prob > 0.0
            and self.transform_args.random_iou_crop.is_active(step)
        )

    def _is_mosaic_active_at_step(self, step: int) -> bool:
        if (
            self.mosaic is None
            or self.transform_args.mosaic is None
            or self.transform_args.mosaic.prob <= 0.0
        ):
            return False
        return self.transform_args.mosaic.is_active(step)

    def _should_apply_mosaic(self) -> bool:
        return (
            self._is_mosaic_active_at_step(self._step)
            and np.random.random() < self.transform_args.mosaic.prob  # type: ignore[union-attr]
        )

    def _get_transform_active_status_at_step(
        self, step: int
    ) -> tuple[bool, bool, bool, bool]:
        return (
            self._is_photometric_distort_active_at_step(step),
            self._is_random_zoom_out_active_at_step(step),
            self._is_random_iou_crop_active_at_step(step),
            self._is_mosaic_active_at_step(step),
        )

    def _is_step_start_or_stop_configured(
        self, activation_policy: ActivationPolicyArgs | None
    ) -> bool:
        if activation_policy is None:
            return False
        return (
            activation_policy.step_start != 0 or activation_policy.step_stop is not None
        )

    def _get_transform_cache_key(
        self, *, step: int, skip_zoomout_ioucrop: bool
    ) -> tuple[bool, bool, bool]:
        return (
            self._is_photometric_distort_active_at_step(step),
            (
                self._is_random_zoom_out_active_at_step(step)
                and not skip_zoomout_ioucrop
            ),
            (
                self._is_random_iou_crop_active_at_step(step)
                and not skip_zoomout_ioucrop
            ),
        )

    def _build_transform(self, key: tuple[bool, bool, bool]) -> Compose:
        photometric_distort_active, random_zoom_out_active, random_iou_crop_active = key
        transforms: list[Any] = []

        if self.channel_drop is not None:
            transforms.append(self.channel_drop)
        if photometric_distort_active:
            transforms.append(self.photometric_distort)  # type: ignore[arg-type]
        if random_zoom_out_active:
            transforms.append(self.random_zoom_out)  # type: ignore[arg-type]
        if random_iou_crop_active:
            transforms.append(self.random_iou_crop)  # type: ignore[arg-type]
        if self.horizontal_flip is not None:
            transforms.append(self.horizontal_flip)
        if self.vertical_flip is not None:
            transforms.append(self.vertical_flip)
        if self.random_rotate_90 is not None:
            transforms.append(self.random_rotate_90)
        if self.random_rotate is not None:
            transforms.append(self.random_rotate)
        if self.resize is not None:
            transforms.append(self.resize)
        transforms.append(self.to_float)
        if self.normalize is not None:
            transforms.append(self.normalize)

        return Compose(
            transforms,
            bbox_params=self.transform_args.bbox_params,
        )

    def _get_transform_from_cache(self, *, skip_zoomout_ioucrop: bool) -> Compose:
        cache_key = self._get_transform_cache_key(
            step=self._step,
            skip_zoomout_ioucrop=skip_zoomout_ioucrop,
        )
        if cache_key not in self._compose_cache:
            # Mosaic samples need the current active pipeline without
            # RandomZoomOut/RandomIoUCrop, so keep that variant cached too.
            self._compose_cache[cache_key] = self._build_transform(cache_key)
        return self._compose_cache[cache_key]

    def set_step(self, step: int) -> None:
        self._step = step

    def uses_step_dependent_worker_state(self) -> bool:
        return (
            (
                self.transform_args.photometric_distort is not None
                and self.transform_args.photometric_distort.prob > 0.0
                and self._is_step_start_or_stop_configured(
                    self.transform_args.photometric_distort
                )
            )
            or (
                self.transform_args.random_zoom_out is not None
                and self.transform_args.random_zoom_out.prob > 0.0
                and self._is_step_start_or_stop_configured(
                    self.transform_args.random_zoom_out
                )
            )
            or (
                self.transform_args.random_iou_crop is not None
                and self.transform_args.random_iou_crop.prob > 0.0
                and self._is_step_start_or_stop_configured(
                    self.transform_args.random_iou_crop
                )
            )
            or (
                self.transform_args.mosaic is not None
                and self.transform_args.mosaic.prob > 0.0
                and self._is_step_start_or_stop_configured(self.transform_args.mosaic)
            )
        )

    def requires_dataloader_reinitialization(self) -> bool:
        return (
            self._get_transform_active_status_at_step(self._step)
            != self._current_transform_active_status
        )

    def mark_dataloader_as_reinitialized(self) -> None:
        self._current_transform_active_status = (
            self._get_transform_active_status_at_step(self._step)
        )

    def __call__(
        self, input: ObjectDetectionTransformInput
    ) -> ObjectDetectionTransformOutput:
        image = input["image"]
        bboxes = input["bboxes"]
        class_labels = input["class_labels"]

        if self._should_apply_mosaic():
            image, bboxes, class_labels = self.mosaic(image, bboxes, class_labels)  # type: ignore[misc]

            # MosaicTransform clips boxes to the canvas but keeps degenerate boxes
            # (zero width/height). Filter them before passing to albumentations.
            if len(bboxes) > 0:
                valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
                bboxes = bboxes[valid]
                class_labels = class_labels[valid]

            transform = self._get_transform_from_cache(skip_zoomout_ioucrop=True)
        else:
            transform = self._get_transform_from_cache(skip_zoomout_ioucrop=False)

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        # Some albumentations versions return lists of tuples instead of arrays.
        bboxes = transformed["bboxes"]
        class_labels = transformed["class_labels"]
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)
        if isinstance(class_labels, list):
            class_labels = np.array(class_labels)

        return {
            "image": transformed["image"],
            "bboxes": bboxes,
            "class_labels": class_labels,
        }


class ObjectDetectionCollateFunction(TaskCollateFunction):
    def __init__(
        self,
        split: Literal["train", "val"],
        transform_args: ObjectDetectionTransformArgs,
    ):
        super().__init__(split, transform_args)
        self.transform_args: ObjectDetectionTransformArgs = transform_args

        self.scale_jitter: BatchReplayCompose | None = None
        self.mixup: MixUp | None = None
        self.copyblend: CopyBlend | None = None

        if self.transform_args.mixup is not None:
            self.mixup = MixUp()

        if self.transform_args.copyblend is not None:
            self.copyblend = CopyBlend(
                area_threshold=self.transform_args.copyblend.area_threshold,
                num_objects=self.transform_args.copyblend.num_objects,
                expand_ratios=self.transform_args.copyblend.expand_ratios,
            )

        if self.transform_args.scale_jitter is not None:
            self.scale_jitter = BatchReplayCompose(
                transforms=[
                    ScaleJitter(
                        sizes=self.transform_args.scale_jitter.sizes,
                        target_size=(
                            no_auto(self.transform_args.image_size)
                            if self.transform_args.scale_jitter.sizes is None
                            else None
                        ),
                        scale_range=self.transform_args.scale_jitter.scale_range,
                        num_scales=self.transform_args.scale_jitter.num_scales,
                        divisible_by=self.transform_args.scale_jitter.divisible_by,
                        p=self.transform_args.scale_jitter.prob,
                    )
                ],
                bbox_params=self.transform_args.bbox_params,
            )

        self._step = 0
        self._current_transform_active_status = (
            self._get_transform_active_status_at_step(self._step)
        )

        self.to_tensor = BatchTransform(
            Compose(
                transforms=[ToTensorV2()],
                bbox_params=self.transform_args.bbox_params,
            )
        )

    def _is_mixup_active_at_step(self, step: int) -> bool:
        if (
            self.mixup is None
            or self.transform_args.mixup is None
            or self.transform_args.mixup.prob <= 0.0
        ):
            return False
        return self.transform_args.mixup.is_active(step)

    def _is_copyblend_active_at_step(self, step: int) -> bool:
        if (
            self.copyblend is None
            or self.transform_args.copyblend is None
            or self.transform_args.copyblend.prob <= 0.0
        ):
            return False
        return self.transform_args.copyblend.is_active(step)

    def _is_scale_jitter_active_at_step(self, step: int) -> bool:
        if (
            self.scale_jitter is None
            or self.transform_args.scale_jitter is None
            or self.transform_args.scale_jitter.prob
            <= 0.0  # TODO (Yutong, 04/26): there is never a scale jitter prob used in LTDETR. Remove it.
        ):
            return False
        return self.transform_args.scale_jitter.is_active(step)

    def _should_apply_mixup(self) -> bool:
        return (
            self.transform_args.mixup is not None
            and self._is_mixup_active_at_step(self._step)
            and np.random.random() < self.transform_args.mixup.prob
        )

    def _should_apply_copyblend(self) -> bool:
        return (
            self.transform_args.copyblend is not None
            and self._is_copyblend_active_at_step(self._step)
            and np.random.random() < self.transform_args.copyblend.prob
        )

    def _get_transform_active_status_at_step(
        self, step: int
    ) -> tuple[bool, bool, bool]:
        return (
            self._is_scale_jitter_active_at_step(step),
            self._is_mixup_active_at_step(step),
            self._is_copyblend_active_at_step(step),
        )

    def set_step(self, step: int) -> None:
        self._step = step

    def uses_step_dependent_worker_state(self) -> bool:
        return (
            (
                self.scale_jitter is not None
                and self.transform_args.scale_jitter is not None
                and self.transform_args.scale_jitter.prob
                > 0.0  # TODO (Yutong, 04/26): there is never a scale jitter prob used in LTDETR. Remove it.
                and self.transform_args.scale_jitter.step_stop is not None
            )
            or (
                self.mixup is not None
                and self.transform_args.mixup is not None
                and self.transform_args.mixup.prob > 0.0
                and (
                    self.transform_args.mixup.step_start != 0
                    or self.transform_args.mixup.step_stop is not None
                )
            )
            or (
                self.copyblend is not None
                and self.transform_args.copyblend is not None
                and self.transform_args.copyblend.prob > 0.0
                and (
                    self.transform_args.copyblend.step_start != 0
                    or self.transform_args.copyblend.step_stop is not None
                )
            )
        )

    def requires_dataloader_reinitialization(self) -> bool:
        return (
            self._get_transform_active_status_at_step(self._step)
            != self._current_transform_active_status
        )

    def mark_dataloader_as_reinitialized(self) -> None:
        self._current_transform_active_status = (
            self._get_transform_active_status_at_step(self._step)
        )

    def __call__(self, batch: list[ObjectDetectionDatasetItem]) -> ObjectDetectionBatch:
        augment_batch = [
            {
                "image": item["image"],
                "bboxes": item["bboxes"],
                "class_labels": item["classes"],
            }
            for item in batch
        ]

        if (
            self.mixup is not None
            and len(augment_batch) >= 2
            and self._should_apply_mixup()
        ):
            augment_batch = self.mixup(batch=augment_batch)
        elif (
            self.copyblend is not None
            and len(augment_batch) > 0
            and self._should_apply_copyblend()
        ):
            # CopyBlend currently operates on bounding boxes as normalized YOLO
            # coordinates in (cx, cy, w, h) format.
            augment_batch = self.copyblend(batch=augment_batch)

        if self.scale_jitter is not None and self._is_scale_jitter_active_at_step(
            self._step
        ):
            augment_batch = self.scale_jitter(batch=augment_batch)

        augment_batch = self.to_tensor(augment_batch)

        for item in augment_batch:
            # Some albumentations versions return lists of tuples instead of arrays.
            if isinstance(item["bboxes"], list):
                item["bboxes"] = np.array(item["bboxes"])
            if isinstance(item["class_labels"], list):
                item["class_labels"] = np.array(item["class_labels"])

        image = torch.stack([item["image"] for item in augment_batch])  # type: ignore
        # Albumentations ToTensorV2 only converts images/masks to tensors. We have to
        # convert the remaining items manually.
        bboxes = [torch.from_numpy(item["bboxes"]).float() for item in augment_batch]
        classes = [
            torch.from_numpy(item["class_labels"]).long() for item in augment_batch
        ]

        out = ObjectDetectionBatch(
            image_path=[item["image_path"] for item in batch],
            image=image,
            bboxes=bboxes,
            classes=classes,
            original_size=[item["original_size"] for item in batch],
        )
        return out
