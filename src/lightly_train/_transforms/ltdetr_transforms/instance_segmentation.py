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
from torch import Tensor

from lightly_train._transforms.ltdetr_transforms.components import (
    StepActivationTracker,
    StepScheduledCompose,
)
from lightly_train._transforms.ltdetr_transforms.utils import (
    filter_boxes_below_min_size,
    filter_degenerate_yolo_boxes,
)
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


class LTDETRInstanceSegmentationTransformArgs(TaskTransformArgs):
    """Transform arguments for LT-DETR instance segmentation.

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


class _MaskAwareStepScheduledCompose(StepScheduledCompose):
    """Step-scheduled compose that also converts images and masks to tensors.

    Instance segmentation appends ``ToTensorV2`` so albumentations converts the
    image and the per-instance masks together, keeping them aligned.
    """

    def _build_transform(self, key: tuple[bool, bool, bool]) -> Compose:
        transform = super()._build_transform(key)
        transform.transforms.append(ToTensorV2())
        return transform


class LTDETRInstanceSegmentationTransform(TaskTransform):
    transform_args_cls: type[LTDETRInstanceSegmentationTransformArgs] = (
        LTDETRInstanceSegmentationTransformArgs
    )

    def __init__(
        self,
        transform_args: LTDETRInstanceSegmentationTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)
        self.transform_args: LTDETRInstanceSegmentationTransformArgs = transform_args
        # The step-scheduling and Compose-caching machinery is a component the
        # transform holds, not a base class it inherits from.
        self._transform_compose = _MaskAwareStepScheduledCompose(transform_args)

    def set_step(self, step: int) -> None:
        self._transform_compose.set_step(step)

    def uses_step_dependent_worker_state(self) -> bool:
        return self._transform_compose.uses_step_dependent_worker_state()

    def requires_dataloader_reinitialization(self) -> bool:
        return self._transform_compose.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self._transform_compose.mark_dataloader_as_reinitialized()

    def __call__(
        self, input: LTDETRInstanceSegmentationTransformInput
    ) -> LTDETRInstanceSegmentationTransformOutput:
        image = input["image"]
        bboxes = input["bboxes"]
        class_labels = input["class_labels"]
        binary_masks = input["binary_masks"]
        indices = np.arange(len(bboxes), dtype=np.int64)

        if self._transform_compose.should_apply_mosaic():
            if image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(
                    "LT-DETR instance segmentation mosaic only supports RGB images "
                    "with shape (H, W, 3). Disable mosaic for non-RGB inputs."
                )
            image, bboxes, class_labels, binary_masks_opt = (
                self._transform_compose.mosaic(  # type: ignore[misc]
                    image, bboxes, class_labels, binary_masks
                )
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
            transform = self._transform_compose.get_transform(skip_zoomout_ioucrop=True)
        else:
            transform = self._transform_compose.get_transform(
                skip_zoomout_ioucrop=len(bboxes) == 0
            )

        if len(binary_masks) == 0:
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
                indices=indices,
            )
            image_out = transformed["image"]
            height, width = image_out.shape[-2:]
            bboxes_out = transformed["bboxes"]
            class_labels_out = transformed["class_labels"]
            if isinstance(bboxes_out, list):
                bboxes_out = np.array(bboxes_out, dtype=np.float64).reshape(-1, 4)
            elif bboxes_out.size == 0:
                bboxes_out = bboxes_out.reshape(0, 4)
            if isinstance(class_labels_out, list):
                class_labels_out = np.array(class_labels_out)
            bboxes_out, class_labels_out, _ = filter_boxes_below_min_size(
                bboxes=bboxes_out,
                class_labels=class_labels_out,
                image_size=(height, width),
                min_size_px=float(self.transform_args.min_bbox_size_px),
            )
            return {
                "image": image_out,
                "binary_masks": image_out.new_zeros(0, height, width, dtype=torch.int),
                "bboxes": bboxes_out,
                "class_labels": class_labels_out,
            }

        transformed = transform(
            image=image,
            masks=binary_masks,
            bboxes=bboxes,
            class_labels=class_labels,
            indices=indices,
        )

        bboxes_out = transformed["bboxes"]
        class_labels_out = transformed["class_labels"]
        indices_out = transformed["indices"]
        if isinstance(bboxes_out, list):
            bboxes_out = np.array(bboxes_out, dtype=np.float64).reshape(-1, 4)
        elif bboxes_out.size == 0:
            bboxes_out = bboxes_out.reshape(0, 4)
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

        # Drop boxes whose final pixel-side width/height is below the configured
        # threshold; re-index masks with the same filtered indices so the three
        # arrays stay aligned.
        bboxes_out, class_labels_out, indices_out = filter_boxes_below_min_size(
            bboxes=bboxes_out,
            class_labels=class_labels_out,
            indices=indices_out,
            image_size=(height, width),
            min_size_px=float(self.transform_args.min_bbox_size_px),
        )
        if indices_out is None:
            # ``indices_out`` should never come back as None from
            # ``filter_boxes_below_min_size`` because we passed ``indices`` in,
            # but fall back defensively.
            kept_binary_masks = binary_masks_out
        elif len(indices_out) == 0:
            kept_binary_masks = image_out.new_zeros(0, height, width, dtype=torch.int)
        else:
            kept_binary_masks = binary_masks_out[torch.from_numpy(indices_out).long()]

        return {
            "image": image_out,
            "binary_masks": kept_binary_masks,
            "bboxes": bboxes_out,
            "class_labels": class_labels_out,
        }


class LTDETRInstanceSegmentationCollateFunction(TaskCollateFunction):
    def __init__(
        self,
        split: Literal["train", "val"],
        transform_args: LTDETRInstanceSegmentationTransformArgs,
    ) -> None:
        super().__init__(split, transform_args)
        self.transform_args: LTDETRInstanceSegmentationTransformArgs = transform_args
        # Reinit bookkeeping is a shared primitive the collate holds, not inherited.
        self._activation = StepActivationTracker(
            self._get_transform_active_status_at_step
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

    def set_step(self, step: int) -> None:
        self._activation.set_step(step)

    def requires_dataloader_reinitialization(self) -> bool:
        return self._activation.requires_dataloader_reinitialization()

    def mark_dataloader_as_reinitialized(self) -> None:
        self._activation.mark_dataloader_as_reinitialized()

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
