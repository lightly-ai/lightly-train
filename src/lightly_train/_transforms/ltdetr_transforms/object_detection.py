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
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.batch_transform import BatchReplayCompose, BatchTransform
from lightly_train._transforms.copyblend import CopyBlend
from lightly_train._transforms.ltdetr_transforms.base import (
    LTDETRTransformArgs,
    _LTDETRCollateFunction,
    _LTDETRTransform,
)
from lightly_train._transforms.ltdetr_transforms.utils import (
    filter_degenerate_yolo_boxes,
    normalize_bboxes_and_labels,
)
from lightly_train._transforms.mixup import MixUp
from lightly_train._transforms.scale_jitter import ScaleJitter
from lightly_train._transforms.task_transform import (
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train.types import (
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


class LTDETRObjectDetectionTransformArgs(LTDETRTransformArgs):
    pass


class LTDETRObjectDetectionTransform(_LTDETRTransform):
    transform_args_cls: type[LTDETRObjectDetectionTransformArgs] = (
        LTDETRObjectDetectionTransformArgs
    )

    def __call__(
        self, input: LTDETRObjectDetectionTransformInput
    ) -> LTDETRObjectDetectionTransformOutput:
        image = input["image"]
        bboxes = input["bboxes"]
        class_labels = input["class_labels"]

        if self._should_apply_mosaic():
            image, bboxes, class_labels, _ = self.mosaic(image, bboxes, class_labels)  # type: ignore[misc]

            # MosaicTransform clips boxes to the canvas but keeps degenerate boxes
            # (zero width/height). Filter them before passing to albumentations.
            bboxes, class_labels, _ = filter_degenerate_yolo_boxes(
                bboxes=bboxes, class_labels=class_labels
            )

            transform = self._get_transform_from_cache(skip_zoomout_ioucrop=True)
        else:
            transform = self._get_transform_from_cache(skip_zoomout_ioucrop=False)

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        bboxes, class_labels = normalize_bboxes_and_labels(
            transformed["bboxes"], transformed["class_labels"]
        )

        return {
            "image": transformed["image"],
            "bboxes": bboxes,
            "class_labels": class_labels,
        }


class LTDETRObjectDetectionCollateFunction(_LTDETRCollateFunction):
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
