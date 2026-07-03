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
from torch import Tensor

from lightly_train._transforms.ltdetr_transforms.base import (
    LTDETRTransformArgs,
    _LTDETRCollateFunction,
    _LTDETRTransform,
)
from lightly_train._transforms.ltdetr_transforms.utils import (
    filter_degenerate_yolo_boxes,
)
from lightly_train._transforms.task_transform import (
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train.types import (
    InstanceSegmentationBatch,
    InstanceSegmentationDatasetItem,
    NDArrayBBoxes,
    NDArrayBinaryMasksInt,
    NDArrayClasses,
    NDArrayImage,
)


class LTDETRInstanceSegmentationTransformInput(TaskTransformInput):
    image: NDArrayImage
    binary_masks: NDArrayBinaryMasksInt
    bboxes: NDArrayBBoxes
    class_labels: NDArrayClasses


class LTDETRInstanceSegmentationTransformOutput(TaskTransformOutput):
    image: Tensor
    binary_masks: Tensor
    bboxes: NDArrayBBoxes
    class_labels: NDArrayClasses


class LTDETRInstanceSegmentationTransformArgs(LTDETRTransformArgs):
    pass


class LTDETRInstanceSegmentationTransform(_LTDETRTransform):
    transform_args_cls: type[LTDETRInstanceSegmentationTransformArgs] = (
        LTDETRInstanceSegmentationTransformArgs
    )

    transform_args: LTDETRInstanceSegmentationTransformArgs

    def _build_transform(self, key: tuple[bool, bool, bool]) -> Compose:
        transform = super()._build_transform(key)
        transform.transforms.append(ToTensorV2())
        return transform

    def __call__(
        self, input: LTDETRInstanceSegmentationTransformInput
    ) -> LTDETRInstanceSegmentationTransformOutput:
        image = input["image"]
        bboxes = input["bboxes"]
        class_labels = input["class_labels"]
        binary_masks = input["binary_masks"]
        indices = np.arange(len(bboxes), dtype=np.int64)

        if self._should_apply_mosaic():
            image, bboxes, class_labels, binary_masks_opt = self.mosaic(  # type: ignore[misc]
                image, bboxes, class_labels, binary_masks
            )
            if binary_masks_opt is None:
                raise RuntimeError("Expected mask-aware mosaic output.")
            binary_masks = binary_masks_opt
            indices = np.arange(len(bboxes), dtype=np.int64)
            bboxes, class_labels, indices_opt = filter_degenerate_yolo_boxes(
                bboxes=bboxes,
                class_labels=class_labels,
                indices=indices,
            )
            assert indices_opt is not None
            indices = indices_opt
            binary_masks = binary_masks[indices]
            indices = np.arange(len(bboxes), dtype=np.int64)
            transform = self._get_transform_from_cache(skip_zoomout_ioucrop=True)
        else:
            transform = self._get_transform_from_cache(skip_zoomout_ioucrop=False)

        transformed = transform(
            image=image,
            masks=[] if len(binary_masks) == 0 else binary_masks,
            bboxes=bboxes,
            class_labels=class_labels,
            indices=indices,
        )

        bboxes_out = transformed["bboxes"]
        class_labels_out = transformed["class_labels"]
        indices_out = transformed["indices"]
        if isinstance(bboxes_out, list):
            bboxes_out = np.array(bboxes_out)
        if isinstance(class_labels_out, list):
            class_labels_out = np.array(class_labels_out)
        if isinstance(indices_out, list):
            indices_out = np.array(indices_out)
        indices_out = indices_out.astype(np.int64, copy=False)

        image_out = transformed["image"]
        masks = transformed["masks"]
        height, width = image_out.shape[-2:]
        # ``masks`` is never filtered by albumentations, so its length can be
        # non-zero even when every box was dropped. Guard on the surviving
        # instances (``indices_out``) instead to avoid ``torch.stack([])``.
        kept_masks = [masks[int(index)] for index in indices_out]
        if len(kept_masks) == 0:
            binary_masks_out = image_out.new_zeros(0, height, width, dtype=torch.int)
        else:
            binary_masks_out = torch.stack(kept_masks).to(dtype=torch.int)

        return {
            "image": image_out,
            "binary_masks": binary_masks_out,
            "bboxes": bboxes_out,
            "class_labels": class_labels_out,
        }


class LTDETRInstanceSegmentationCollateFunction(_LTDETRCollateFunction):
    def __init__(
        self,
        split: Literal["train", "val"],
        transform_args: LTDETRInstanceSegmentationTransformArgs,
    ) -> None:
        super().__init__(split, transform_args)
        self.transform_args: LTDETRInstanceSegmentationTransformArgs = transform_args
        self._step = 0
        self._current_transform_active_status = (
            self._get_transform_active_status_at_step(self._step)
        )

    def _get_transform_active_status_at_step(self, step: int) -> tuple[bool]:
        return (self._is_mixup_active_at_step(step),)

    def uses_step_dependent_worker_state(self) -> bool:
        return (
            self.transform_args.mixup is not None
            and self.transform_args.mixup.prob > 0.0
            and (
                self.transform_args.mixup.step_start != 0
                or self.transform_args.mixup.step_stop is not None
            )
        )

    def __call__(
        self, batch: list[InstanceSegmentationDatasetItem]
    ) -> InstanceSegmentationBatch:
        images = [item["image"] for item in batch]
        bboxes = [item["bboxes"] for item in batch]
        classes = [item["classes"] for item in batch]
        masks = [item["binary_masks"]["masks"] for item in batch]

        if self.split == "train":
            image_batch_train = torch.stack(images)
            if len(batch) >= 2 and self._should_apply_mixup():
                beta = torch.empty(1).uniform_(0.45, 0.55).item()
                shifted_images = image_batch_train.roll(shifts=1, dims=0)
                image_batch_train = image_batch_train.mul(beta).add(
                    shifted_images.mul(1.0 - beta)
                )
                bboxes = [
                    torch.cat([current, shifted], dim=0)
                    for current, shifted in zip(bboxes, bboxes[-1:] + bboxes[:-1])
                ]
                classes = [
                    torch.cat([current, shifted], dim=0)
                    for current, shifted in zip(classes, classes[-1:] + classes[:-1])
                ]
                masks = [
                    torch.cat([current, shifted], dim=0)
                    for current, shifted in zip(masks, masks[-1:] + masks[:-1])
                ]
            image_batch: Tensor | list[Tensor] = image_batch_train
        else:
            image_batch = images

        return {
            "image_path": [item["image_path"] for item in batch],
            "image": image_batch,
            "binary_masks": [
                {"masks": mask.bool(), "labels": labels}
                for mask, labels in zip(masks, classes)
            ],
            "bboxes": bboxes,
            "classes": classes,
        }
