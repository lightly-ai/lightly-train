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
from albumentations import BboxParams, Compose
from torch import Tensor

from lightly_train._transforms.eomt_transforms.utils import (
    _build_transforms_eomt,
    _resolve_incompatible_eomt,
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
    ColorJitterArgs,
    NormalizeArgs,
    RandomCropArgs,
    RandomFlipArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    ScaleJitterArgs,
    SmallestMaxSizeArgs,
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


class EoMTInstanceSegmentationTransformInput(TaskTransformInput):
    image: NDArrayImage
    binary_masks: NDArrayBinaryMasksInt
    bboxes: NDArrayBBoxes
    class_labels: NDArrayClasses


class EoMTInstanceSegmentationTransformOutput(TaskTransformOutput):
    image: Tensor
    binary_masks: Tensor
    bboxes: NDArrayBBoxes
    class_labels: NDArrayClasses


class EoMTInstanceSegmentationTransformArgs(TaskTransformArgs):
    image_size: ImageSizeTuple | Literal["auto"] | None
    channel_drop: ChannelDropArgs | None
    num_channels: int | Literal["auto"]
    normalize: NormalizeArgs | Literal["auto"]
    random_flip: RandomFlipArgs | None
    random_rotate_90: RandomRotate90Args | None
    random_rotate: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None
    # TODO: Lionel(09/25): These are currently not fully used.
    scale_jitter: ScaleJitterArgs | None
    smallest_max_size: SmallestMaxSizeArgs | None
    random_crop: RandomCropArgs | None
    bbox_params: BboxParams

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        pass

    def resolve_incompatible(self) -> None:
        _resolve_incompatible_eomt(transform_args=self)


class EoMTInstanceSegmentationTransform(TaskTransform):
    transform_args_cls: type[EoMTInstanceSegmentationTransformArgs] = (
        EoMTInstanceSegmentationTransformArgs
    )

    def __init__(
        self,
        transform_args: EoMTInstanceSegmentationTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform = Compose(
            _build_transforms_eomt(transform_args=transform_args),
            bbox_params=transform_args.bbox_params,
        )

    def __call__(
        self, input: EoMTInstanceSegmentationTransformInput
    ) -> EoMTInstanceSegmentationTransformOutput:
        # Handle the case when there are no masks. Albumentations doesn't like passing
        # empty numpy arrays as masks.
        if len(input["binary_masks"]) == 0:
            # Transform only the image without masks.
            transformed = self.transform(
                image=input["image"],
                # Pass empty lists to avoid albumentations errors. Yes this is weird,
                # empty numpy arrays don't work but empty lists do.
                masks=[],
                bboxes=[],
                class_labels=[],
                indices=[],
            )
            image = transformed["image"]
            H, W = image.shape[-2:]
            return {
                "image": image,
                "binary_masks": image.new_zeros(0, H, W, dtype=torch.int),
                "bboxes": np.array([], dtype=np.float64).reshape(0, 4),
                "class_labels": np.array([], dtype=np.int64),
            }

        # Mask augmentations only work correctly when passed as `masks` to albumentations.
        # Passing as `binary_masks` and adding `additional_targets={"binary_masks": "masks"}`
        # doesn't work. "mask" also doesn't work as target.
        transformed = self.transform(
            image=input["image"],
            masks=input["binary_masks"],
            bboxes=input["bboxes"],
            class_labels=input["class_labels"],
            indices=np.arange(len(input["bboxes"])),
        )

        # Albumentations can drop bboxes if they are out of the image after the transform.
        # It also automatically drops the corresponding class labels and indices but
        # this doesn't work for masks. So we need to filter them out manually.
        masks = transformed["masks"]
        masks = [masks[i] for i in transformed["indices"]]
        image = transformed["image"]
        H, W = image.shape[-2:]
        binary_masks = (
            torch.stack(masks)
            if len(masks) > 0
            else image.new_zeros(0, H, W, dtype=torch.int)
        )

        return {
            "image": transformed["image"],
            "binary_masks": binary_masks,
            "bboxes": transformed["bboxes"],
            "class_labels": transformed["class_labels"],
        }


class EoMTInstanceSegmentationCollateFunction(TaskCollateFunction):
    def __call__(
        self, batch: list[InstanceSegmentationDatasetItem]
    ) -> InstanceSegmentationBatch:
        # Prepare the batch without any stacking.
        images = [item["image"] for item in batch]

        out: InstanceSegmentationBatch = {
            "image_path": [item["image_path"] for item in batch],
            # Stack images during training as they all have the same shape.
            # During validation every image can have a different shape.
            "image": torch.stack(images) if self.split == "train" else images,
            "binary_masks": [item["binary_masks"] for item in batch],
            "bboxes": [item["bboxes"] for item in batch],
            "classes": [item["classes"] for item in batch],
        }

        return out
