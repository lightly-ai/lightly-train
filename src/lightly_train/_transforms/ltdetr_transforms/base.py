#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from albumentations import BboxParams, Compose
from pydantic import ConfigDict

from lightly_train._transforms.ltdetr_transforms.utils import (
    build_ltdetr_sample_transform_parts,
    is_step_start_or_stop_configured,
    ordered_ltdetr_sample_transforms,
)
from lightly_train._transforms.mosaic import MosaicTransform
from lightly_train._transforms.task_transform import (
    TaskCollateFunction,
    TaskTransform,
    TaskTransformArgs,
)
from lightly_train._transforms.transform import (
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
from lightly_train.types import ImageSizeTuple


class LTDETRTransformArgs(TaskTransformArgs):
    """Shared transform arguments for LT-DETR-style tasks (object detection,
    instance segmentation, oriented object detection).

    Task-specific transforms subclass this to set their own field defaults; it is
    not used directly.
    """

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


class _LTDETRTransform(TaskTransform):
    """Shared step-scheduling and Compose-caching machinery for LT-DETR sample
    transforms.

    Subclasses (object detection, instance segmentation) implement ``__call__`` and,
    if needed, override ``_build_transform``. This is internal scaffolding and is not
    meant to be instantiated directly.
    """

    transform_args_cls: type[LTDETRTransformArgs] = LTDETRTransformArgs

    def __init__(
        self,
        transform_args: LTDETRTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform_args: LTDETRTransformArgs = transform_args
        self._step = 0
        self._transform_parts = build_ltdetr_sample_transform_parts(transform_args)
        self.photometric_distort = self._transform_parts["photometric_distort"]
        self.random_zoom_out = self._transform_parts["random_zoom_out"]
        self.random_iou_crop = self._transform_parts["random_iou_crop"]

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
        transforms = ordered_ltdetr_sample_transforms(
            self._transform_parts,
            photometric_distort_active=photometric_distort_active,
            random_zoom_out_active=random_zoom_out_active,
            random_iou_crop_active=random_iou_crop_active,
        )

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
                and is_step_start_or_stop_configured(
                    self.transform_args.photometric_distort
                )
            )
            or (
                self.transform_args.random_zoom_out is not None
                and self.transform_args.random_zoom_out.prob > 0.0
                and is_step_start_or_stop_configured(
                    self.transform_args.random_zoom_out
                )
            )
            or (
                self.transform_args.random_iou_crop is not None
                and self.transform_args.random_iou_crop.prob > 0.0
                and is_step_start_or_stop_configured(
                    self.transform_args.random_iou_crop
                )
            )
            or (
                self.transform_args.mosaic is not None
                and self.transform_args.mosaic.prob > 0.0
                and is_step_start_or_stop_configured(self.transform_args.mosaic)
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


# Shared LT-DETR augmentation presets. These carry the default augmentation
# hyperparameters used by both object detection and instance segmentation (the two
# tasks share the same augmentation pipeline, matching the upstream ECDet/ECSeg configs).
class LTDETRRandomPhotometricDistortArgs(RandomPhotometricDistortArgs):
    prob: float = 0.5

    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means photometric distort is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRRandomZoomOutArgs(RandomZoomOutArgs):
    prob: float = 0.5

    fill: float = 0.0
    side_range: tuple[float, float] = (1.0, 4.0)

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means random zoom out is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRRandomIoUCropArgs(RandomIoUCropArgs):
    prob: float = 0.8

    min_scale: float = 0.3
    max_scale: float = 1.0
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    sampler_options: Sequence[float] | None = None
    crop_trials: int = 40
    iou_trials: int = 1000

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" resolves to epoch total_epochs - no_aug_epoch. For shorter runs,
    # no_aug_epoch is scaled following a certain rule. See :func:`resolve_ltdetr_step_schedule` for the full algorithm.
    # None means random IoU crop is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRRandomFlipArgs(RandomFlipArgs):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class LTDETRMosaicArgs(MosaicArgs):
    prob: float = 0.5

    output_size: int = 320
    max_size: int | None = None
    rotation_range: float = 10.0
    translation_range: tuple[float, float] = (0.1, 0.1)
    scaling_range: tuple[float, float] = (0.5, 1.5)
    fill_value: int | float = 0
    max_cached_images: int = 50
    random_pop: bool = True

    # "auto" resolves to epoch 4, or to floor(total_epochs / 3) for runs
    # with <= 12 epochs.
    step_start: int | Literal["auto"] = "auto"
    # "auto" uses a compressed short-run schedule for <= 12 epochs and
    # transitions to the midpoint rule on longer runs.
    # None means mosaic is always on.
    step_stop: int | Literal["auto"] | None = "auto"


class LTDETRResizeArgs(ResizeArgs):
    height: int | Literal["auto"] = "auto"
    width: int | Literal["auto"] = "auto"


class _LTDETRCollateFunction(TaskCollateFunction):
    """Shared step-scheduling scaffolding for LT-DETR collate functions.

    Subclasses provide ``_get_transform_active_status_at_step``,
    ``uses_step_dependent_worker_state`` and ``__call__``. This is internal
    scaffolding and is not meant to be instantiated directly.
    """

    transform_args: LTDETRTransformArgs
    _step: int
    _current_transform_active_status: tuple[bool, ...]

    def _get_transform_active_status_at_step(self, step: int) -> tuple[bool, ...]:
        raise NotImplementedError()

    def _is_mixup_active_at_step(self, step: int) -> bool:
        if self.transform_args.mixup is None or self.transform_args.mixup.prob <= 0.0:
            return False
        return self.transform_args.mixup.is_active(step)

    def _should_apply_mixup(self) -> bool:
        return (
            self.transform_args.mixup is not None
            and self._is_mixup_active_at_step(self._step)
            and np.random.random() < self.transform_args.mixup.prob
        )

    def set_step(self, step: int) -> None:
        self._step = step

    def requires_dataloader_reinitialization(self) -> bool:
        return (
            self._get_transform_active_status_at_step(self._step)
            != self._current_transform_active_status
        )

    def mark_dataloader_as_reinitialized(self) -> None:
        self._current_transform_active_status = (
            self._get_transform_active_status_at_step(self._step)
        )
