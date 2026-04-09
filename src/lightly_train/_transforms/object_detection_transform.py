#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

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
    StopPolicyArgs,
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
    stop_policy: StopPolicyArgs | None
    mixup: MixUpArgs | None = None
    copyblend: CopyBlendArgs | None = None
    mosaic: MosaicArgs | None = None
    scale_jitter: ScaleJitterArgs | None
    resize: ResizeArgs | None
    bbox_params: BboxParams | None
    normalize: NormalizeArgs | Literal["auto"] | None

    # Necessary for the StopPolicyArgs, which are not serializable by pydantic.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        pass

    def resolve_incompatible(self) -> None:
        # TODO: Lionel (09/25): Add checks for incompatible args.
        pass


class ObjectDetectionTransform(TaskTransform):
    transform_args_cls: type[ObjectDetectionTransformArgs] = (
        ObjectDetectionTransformArgs
    )
    _MOSAIC_SKIP_TYPES = (RandomZoomOut, RandomIoUCrop)

    def __init__(
        self,
        transform_args: ObjectDetectionTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform_args: ObjectDetectionTransformArgs = transform_args
        self.stop_step = (
            transform_args.stop_policy.stop_step if transform_args.stop_policy else None
        )

        # TODO: Lionel (09/25): Implement stopping of certain augmentations after some steps.
        if self.stop_step is not None:
            raise NotImplementedError(
                "Stopping certain augmentations after some steps is not implemented yet."
            )
        self.global_step = 0  # Currently hardcoded, will be set from outside.
        self.stop_ops = (
            transform_args.stop_policy.ops if transform_args.stop_policy else set()
        )
        self.past_stop = False

        self.individual_transforms = []

        if transform_args.channel_drop is not None:
            self.individual_transforms += [
                ChannelDrop(
                    num_channels_keep=transform_args.channel_drop.num_channels_keep,
                    weight_drop=transform_args.channel_drop.weight_drop,
                )
            ]

        if transform_args.photometric_distort is not None:
            self.individual_transforms += [
                RandomPhotometricDistort(
                    brightness=transform_args.photometric_distort.brightness,
                    contrast=transform_args.photometric_distort.contrast,
                    saturation=transform_args.photometric_distort.saturation,
                    hue=transform_args.photometric_distort.hue,
                    p=transform_args.photometric_distort.prob,
                )
            ]

        if transform_args.random_zoom_out is not None:
            self.individual_transforms += [
                RandomZoomOut(
                    fill=transform_args.random_zoom_out.fill,
                    side_range=transform_args.random_zoom_out.side_range,
                    p=transform_args.random_zoom_out.prob,
                )
            ]

        if transform_args.random_iou_crop is not None:
            self.individual_transforms += [
                RandomIoUCrop(
                    min_scale=transform_args.random_iou_crop.min_scale,
                    max_scale=transform_args.random_iou_crop.max_scale,
                    min_aspect_ratio=transform_args.random_iou_crop.min_aspect_ratio,
                    max_aspect_ratio=transform_args.random_iou_crop.max_aspect_ratio,
                    sampler_options=transform_args.random_iou_crop.sampler_options,
                    crop_trials=transform_args.random_iou_crop.crop_trials,
                    iou_trials=transform_args.random_iou_crop.iou_trials,
                    p=transform_args.random_iou_crop.prob,
                )
            ]

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                self.individual_transforms += [
                    HorizontalFlip(p=transform_args.random_flip.horizontal_prob)
                ]
            if transform_args.random_flip.vertical_prob > 0.0:
                self.individual_transforms += [
                    VerticalFlip(p=transform_args.random_flip.vertical_prob)
                ]

        if transform_args.random_rotate_90 is not None:
            self.individual_transforms += [
                RandomRotate90(p=transform_args.random_rotate_90.prob)
            ]
        if transform_args.random_rotate is not None:
            self.individual_transforms += [
                Rotate(
                    limit=transform_args.random_rotate.degrees,
                    interpolation=transform_args.random_rotate.interpolation,
                    p=transform_args.random_rotate.prob,
                )
            ]

        if transform_args.resize is not None:
            self.individual_transforms += [
                Resize(
                    height=no_auto(transform_args.resize.height),
                    width=no_auto(transform_args.resize.width),
                )
            ]

        # Scale to [0, 1].
        self.individual_transforms += [
            ToFloat(max_value=255.0),
        ]

        # Only used with ViT-S/16, ViT-T/16+ and ViT-T/16.
        if transform_args.normalize is not None:
            self.individual_transforms += [
                Normalize(
                    mean=no_auto(transform_args.normalize).mean,
                    std=no_auto(transform_args.normalize).std,
                    max_pixel_value=1.0,  # Already scaled.
                )
            ]

        self.transform = Compose(
            self.individual_transforms,
            bbox_params=transform_args.bbox_params,
        )

        # Mosaic setup.
        self.individual_transforms_skip_zoomout_ioucrop: list[Any] = []
        self.transform_skip_zoomout_ioucrop: Compose | None = None
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

        # Build skip pipeline (without RandomZoomOut and RandomIoUCrop) for mosaic.
        self._build_skip_pipeline()

        # Step-aware state for mosaic scheduling.
        self._step = 0
        self._current_mosaic_active_status = self._is_mosaic_active_at_step(0)

    def _build_skip_pipeline(self) -> None:
        """Build the skip pipeline (without RandomZoomOut and RandomIoUCrop) for mosaic."""
        if self.mosaic is not None:
            self.individual_transforms_skip_zoomout_ioucrop = [
                t
                for t in self.individual_transforms
                if not isinstance(t, self._MOSAIC_SKIP_TYPES)
            ]
            self.transform_skip_zoomout_ioucrop = Compose(
                self.individual_transforms_skip_zoomout_ioucrop,
                bbox_params=self.transform_args.bbox_params,
            )

    def _is_mosaic_active_at_step(self, step: int) -> bool:
        if (
            self.mosaic is None
            or self.transform_args.mosaic is None
            or self.transform_args.mosaic.prob <= 0.0
        ):
            return False
        return (
            self.transform_args.mosaic.step_start
            <= step
            < self.transform_args.mosaic.step_stop
        )

    def _should_apply_mosaic(self) -> bool:
        return (
            self._is_mosaic_active_at_step(self._step)
            and np.random.random() < self.transform_args.mosaic.prob  # type: ignore[union-attr]
        )

    def set_step(self, step: int) -> None:
        self._step = step

    def uses_step_dependent_worker_state(self) -> bool:
        return (
            self.mosaic is not None
            and self.transform_args.mosaic is not None
            and self.transform_args.mosaic.prob > 0.0
        )

    def requires_dataloader_reinitialization(self) -> bool:
        return (
            self._is_mosaic_active_at_step(self._step)
            != self._current_mosaic_active_status
        )

    def mark_dataloader_as_reinitialized(self) -> None:
        self._current_mosaic_active_status = self._is_mosaic_active_at_step(self._step)

    def __call__(
        self, input: ObjectDetectionTransformInput
    ) -> ObjectDetectionTransformOutput:
        # Adjust transform after stop_step is reached.
        if (
            self.stop_step is not None
            and self.global_step >= self.stop_step
            and not self.past_stop
        ):
            self.individual_transforms = [
                t for t in self.individual_transforms if type(t) not in self.stop_ops
            ]
            self.transform = Compose(
                self.individual_transforms,
                bbox_params=self.transform_args.bbox_params,
            )
            self._build_skip_pipeline()
            self.past_stop = True

        image = input["image"]
        bboxes = input["bboxes"]
        class_labels = input["class_labels"]

        if self._should_apply_mosaic():
            assert self.mosaic is not None
            assert self.transform_skip_zoomout_ioucrop is not None
            image, bboxes, class_labels = self.mosaic(image, bboxes, class_labels)
            # MosaicTransform clips boxes to the canvas but keeps degenerate boxes
            # (zero width/height). Filter them before passing to albumentations.
            if len(bboxes) > 0:
                valid = (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
                bboxes = bboxes[valid]
                class_labels = class_labels[valid]
            transformed = self.transform_skip_zoomout_ioucrop(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
        else:
            transformed = self.transform(
                image=image, bboxes=bboxes, class_labels=class_labels
            )

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
        return (
            self.transform_args.mixup.step_start
            <= step
            < self.transform_args.mixup.step_stop
        )

    def _is_copyblend_active_at_step(self, step: int) -> bool:
        if (
            self.copyblend is None
            or self.transform_args.copyblend is None
            or self.transform_args.copyblend.prob <= 0.0
        ):
            return False
        return (
            self.transform_args.copyblend.step_start
            <= step
            < self.transform_args.copyblend.step_stop
        )

    def _is_scale_jitter_active_at_step(self, step: int) -> bool:
        if (
            self.scale_jitter is None
            or self.transform_args.scale_jitter is None
            or self.transform_args.scale_jitter.prob
            <= 0.0  # TODO (Yutong, 04/26): there is never a scale jitter prob used in LTDETR. Remove it.
        ):
            return False

        scale_jitter_step_stop = self.transform_args.scale_jitter.step_stop
        if scale_jitter_step_stop is None:
            return True
        return step < scale_jitter_step_stop

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
            )
            or (
                self.copyblend is not None
                and self.transform_args.copyblend is not None
                and self.transform_args.copyblend.prob > 0.0
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
