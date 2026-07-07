#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from albumentations import Compose

from lightly_train._transforms.ltdetr_transforms.utils import (
    build_ltdetr_sample_transform_parts,
    is_step_start_or_stop_configured,
    ordered_ltdetr_sample_transforms,
)
from lightly_train._transforms.mosaic import MosaicTransform

if TYPE_CHECKING:
    from lightly_train._transforms.ltdetr_transforms.object_detection import (
        LTDETRObjectDetectionTransformArgs,
    )


class StepActivationTracker:
    """Tracks whether the set of active step-scheduled augmentations changed since the
    last dataloader (re)initialization.

    Step-aware transforms and collate functions compose this instead of inheriting the
    bookkeeping, so the "did the active set change → reinitialize the loader" logic
    lives in a single place. ``active_status_at_step`` returns the tuple of active
    flags for a given step; a change in that tuple is what requires the dataloader to
    be reinitialized.
    """

    def __init__(
        self, active_status_at_step: Callable[[int], tuple[bool, ...]]
    ) -> None:
        self._active_status_at_step = active_status_at_step
        self._step = 0
        self._current_active_status = active_status_at_step(self._step)

    @property
    def step(self) -> int:
        return self._step

    def set_step(self, step: int) -> None:
        self._step = step

    def requires_dataloader_reinitialization(self) -> bool:
        return self._active_status_at_step(self._step) != self._current_active_status

    def mark_dataloader_as_reinitialized(self) -> None:
        self._current_active_status = self._active_status_at_step(self._step)


class StepScheduledCompose:
    """Step-aware albumentations ``Compose`` for LT-DETR sample transforms.

    Builds the effective per-sample pipeline for the current training step and caches
    it, since the step-scheduled parts only toggle at a few step boundaries. Also owns
    the optional mosaic augmentation. This is a reusable component that sample
    transforms (object detection, and any future LT-DETR task) *hold* rather than
    inherit from.
    """

    def __init__(self, transform_args: LTDETRObjectDetectionTransformArgs) -> None:
        self.transform_args = transform_args
        self._transform_parts = build_ltdetr_sample_transform_parts(transform_args)
        self.photometric_distort = self._transform_parts["photometric_distort"]
        self.random_zoom_out = self._transform_parts["random_zoom_out"]
        self.random_iou_crop = self._transform_parts["random_iou_crop"]

        self._mosaic: MosaicTransform | None = None
        if transform_args.mosaic is not None:
            self._mosaic = MosaicTransform(
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
        self._activation = StepActivationTracker(
            self._get_transform_active_status_at_step
        )

    @property
    def step(self) -> int:
        return self._activation.step

    @property
    def mosaic(self) -> MosaicTransform | None:
        return self._mosaic

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

    def should_apply_mosaic(self) -> bool:
        return (
            self._is_mosaic_active_at_step(self.step)
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

    def get_transform(self, *, skip_zoomout_ioucrop: bool) -> Compose:
        cache_key = self._get_transform_cache_key(
            step=self.step,
            skip_zoomout_ioucrop=skip_zoomout_ioucrop,
        )
        if cache_key not in self._compose_cache:
            # Mosaic samples need the current active pipeline without
            # RandomZoomOut/RandomIoUCrop, so keep that variant cached too.
            self._compose_cache[cache_key] = self._build_transform(cache_key)
        return self._compose_cache[cache_key]

    def set_step(self, step: int) -> None:
        self._activation.set_step(step)

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
        return self._activation.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self._activation.mark_dataloader_as_reinitialized()
