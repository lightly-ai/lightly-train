#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from albumentations import BboxParams, Compose
from albumentations.pytorch.transforms import ToTensorV2
from pydantic import ConfigDict
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.batch_transform import BatchReplayCompose, BatchTransform
from lightly_train._transforms.copyblend import CopyBlend
from lightly_train._transforms.ltdetr_transforms.components import (
    StepActivationTracker,
    StepScheduledCompose,
)
from lightly_train._transforms.ltdetr_transforms.utils import (
    filter_boxes_below_min_size,
    filter_degenerate_yolo_boxes,
    normalize_bboxes_and_labels,
)
from lightly_train._transforms.mixup import MixUp
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
)
from lightly_train.types import (
    ImageSizeTuple,
    NDArrayBBoxes,
    NDArrayClasses,
    NDArrayImage,
    ObjectDetectionBatch,
    ObjectDetectionDatasetItem,
)


class LTDETRObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NotRequired[NDArrayBBoxes]
    class_labels: NotRequired[NDArrayClasses]


class LTDETRObjectDetectionTransformOutput(TaskTransformOutput):
    image: NDArrayImage
    bboxes: NotRequired[NDArrayBBoxes]
    class_labels: NotRequired[NDArrayClasses]


class LTDETRObjectDetectionTransformArgs(TaskTransformArgs):
    """Transform arguments for LT-DETR object detection.

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
    min_bbox_size_px: float = 4.0

    # Necessary for BboxParams, which are not serializable by pydantic.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LTDETRObjectDetectionTransform(TaskTransform):
    transform_args_cls: type[LTDETRObjectDetectionTransformArgs] = (
        LTDETRObjectDetectionTransformArgs
    )

    def __init__(
        self,
        transform_args: LTDETRObjectDetectionTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)
        self.transform_args: LTDETRObjectDetectionTransformArgs = transform_args
        # The step-scheduling and Compose-caching machinery is a component the
        # transform holds, not a base class it inherits from.
        self._transform_compose = StepScheduledCompose(transform_args)

    def set_step(self, step: int) -> None:
        self._transform_compose.set_step(step)

    def uses_step_dependent_worker_state(self) -> bool:
        return self._transform_compose.uses_step_dependent_worker_state()

    def requires_dataloader_reinitialization(self) -> bool:
        return self._transform_compose.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self._transform_compose.mark_dataloader_as_reinitialized()

    def __call__(
        self, input: LTDETRObjectDetectionTransformInput
    ) -> LTDETRObjectDetectionTransformOutput:
        image = input["image"]
        bboxes = input["bboxes"]
        class_labels = input["class_labels"]

        if self._transform_compose.should_apply_mosaic():
            image, bboxes, class_labels, _ = self._transform_compose.mosaic(  # type: ignore[misc]
                image, bboxes, class_labels
            )

            # MosaicTransform clips boxes to the canvas but keeps degenerate boxes
            # (zero width/height). Filter them before passing to albumentations.
            bboxes, class_labels, _ = filter_degenerate_yolo_boxes(
                bboxes=bboxes, class_labels=class_labels
            )

            transform = self._transform_compose.get_transform(skip_zoomout_ioucrop=True)
        else:
            transform = self._transform_compose.get_transform(
                skip_zoomout_ioucrop=False
            )

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        bboxes, class_labels = normalize_bboxes_and_labels(
            transformed["bboxes"], transformed["class_labels"]
        )

        # Drop boxes whose final pixel-side width/height is below the configured
        # threshold. Mirrors the original side guard from the LT-DETR matcher /
        # criterion: tiny targets can destabilize the loss.
        out_image = transformed["image"]
        if out_image.ndim >= 2:
            out_height, out_width = int(out_image.shape[0]), int(out_image.shape[1])
        else:
            out_height, out_width = 0, 0
        bboxes, class_labels, _ = filter_boxes_below_min_size(
            bboxes=bboxes,
            class_labels=class_labels,
            image_size=(out_height, out_width),
            min_size_px=float(self.transform_args.min_bbox_size_px),
        )

        return {
            "image": out_image,
            "bboxes": bboxes,
            "class_labels": class_labels,
        }


class LTDETRObjectDetectionCollateFunction(TaskCollateFunction):
    def __init__(
        self,
        split: Literal["train", "val"],
        transform_args: LTDETRObjectDetectionTransformArgs,
    ):
        super().__init__(split, transform_args)
        self.transform_args: LTDETRObjectDetectionTransformArgs = transform_args

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
                        divisible_by=no_auto(
                            self.transform_args.scale_jitter.divisible_by
                        ),
                        p=self.transform_args.scale_jitter.prob,
                    )
                ],
                bbox_params=self.transform_args.bbox_params,
            )

        # Reinit bookkeeping is a shared primitive the collate holds, not inherited.
        self._activation = StepActivationTracker(
            self._get_transform_active_status_at_step
        )

        self.to_tensor = BatchTransform(
            Compose(
                transforms=[ToTensorV2()],
                bbox_params=self.transform_args.bbox_params,
            )
        )

    def _is_mixup_active_at_step(self, step: int) -> bool:
        if self.transform_args.mixup is None or self.transform_args.mixup.prob <= 0.0:
            return False
        return self.transform_args.mixup.is_active(step)

    def _should_apply_mixup(self) -> bool:
        return (
            self.transform_args.mixup is not None
            and self._is_mixup_active_at_step(self._activation.step)
            and np.random.random() < self.transform_args.mixup.prob
        )

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
            or self.transform_args.scale_jitter.prob <= 0.0
        ):
            return False
        return self.transform_args.scale_jitter.is_active(step)

    def _should_apply_copyblend(self) -> bool:
        return (
            self.transform_args.copyblend is not None
            and self._is_copyblend_active_at_step(self._activation.step)
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

    def uses_step_dependent_worker_state(self) -> bool:
        return (
            (
                self.scale_jitter is not None
                and self.transform_args.scale_jitter is not None
                and self.transform_args.scale_jitter.prob > 0.0
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

    def set_step(self, step: int) -> None:
        self._activation.set_step(step)

    def requires_dataloader_reinitialization(self) -> bool:
        return self._activation.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self._activation.mark_dataloader_as_reinitialized()

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
            self._activation.step
        ):
            augment_batch = self.scale_jitter(batch=augment_batch)

        augment_batch = self.to_tensor(augment_batch)

        for item in augment_batch:
            # Some albumentations versions return lists of tuples instead of arrays.
            if isinstance(item["bboxes"], list):
                item["bboxes"] = np.array(item["bboxes"])
            if isinstance(item["class_labels"], list):
                item["class_labels"] = np.array(item["class_labels"])

            # Drop boxes made too small by the batch-level transforms (mixup,
            # copyblend, scale-jitter, ToTensorV2). Uses the just-finalized image
            # size so the threshold is in the same units as the boxes at this
            # stage.
            img = item["image"]
            if (
                isinstance(img, torch.Tensor)
                and item["bboxes"].size > 0
                and float(self.transform_args.min_bbox_size_px) > 0.0
            ):
                if img.ndim == 3:
                    out_height, out_width = int(img.shape[1]), int(img.shape[2])
                elif img.ndim == 2:
                    out_height, out_width = int(img.shape[0]), int(img.shape[1])
                else:
                    out_height, out_width = 0, 0
                if out_height > 0 and out_width > 0:
                    item["bboxes"], item["class_labels"], _ = (
                        filter_boxes_below_min_size(
                            bboxes=item["bboxes"],
                            class_labels=item["class_labels"],
                            image_size=(out_height, out_width),
                            min_size_px=float(self.transform_args.min_bbox_size_px),
                        )
                    )

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
