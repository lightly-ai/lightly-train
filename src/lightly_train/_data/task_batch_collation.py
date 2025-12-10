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

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.scale_jitter import ScaleJitter
from lightly_train._transforms.task_transform import TaskTransformArgs
from lightly_train.types import (
    InstanceSegmentationBatch,
    InstanceSegmentationDatasetItem,
    MaskPanopticSegmentationBatch,
    MaskPanopticSegmentationDatasetItem,
    MaskSemanticSegmentationBatch,
    MaskSemanticSegmentationDatasetItem,
    ObjectDetectionBatch,
    ObjectDetectionDatasetItem,
)


class BaseCollateFunction:
    def __init__(
        self, split: Literal["train", "val"], transform_args: TaskTransformArgs
    ):
        self.split = split
        self.transform_args = transform_args


class MaskSemanticSegmentationCollateFunction(BaseCollateFunction):
    def __call__(
        self, batch: list[MaskSemanticSegmentationDatasetItem]
    ) -> MaskSemanticSegmentationBatch:
        # Prepare the batch without any stacking.
        images = [item["image"] for item in batch]
        masks = [item["mask"] for item in batch]

        out: MaskSemanticSegmentationBatch = {
            "image_path": [item["image_path"] for item in batch],
            # Stack images during training as they all have the same shape.
            # During validation every image can have a different shape.
            "image": torch.stack(images) if self.split == "train" else images,
            "mask": torch.stack(masks) if self.split == "train" else masks,
            "binary_masks": [item["binary_masks"] for item in batch],
        }

        return out


class InstanceSegmentationCollateFunction(BaseCollateFunction):
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


class MaskPanopticSegmentationCollateFunction(BaseCollateFunction):
    def __call__(
        self, batch: list[MaskPanopticSegmentationDatasetItem]
    ) -> MaskPanopticSegmentationBatch:
        # Prepare the batch without any stacking.
        images = [item["image"] for item in batch]
        masks = [item["masks"] for item in batch]

        out: MaskPanopticSegmentationBatch = {
            "image_path": [item["image_path"] for item in batch],
            # Stack images during training as they all have the same shape.
            # During validation every image can have a different shape.
            "image": torch.stack(images) if self.split == "train" else images,
            "masks": torch.stack(masks) if self.split == "train" else masks,
            "binary_masks": [item["binary_masks"] for item in batch],
        }

        return out


class ObjectDetectionCollateFunction(BaseCollateFunction):
    def __init__(
        self, split: Literal["train", "val"], transform_args: TaskTransformArgs
    ):
        super().__init__(split, transform_args)
        assert isinstance(transform_args, ObjectDetectionTransformArgs)
        self.scale_jitter: Compose | None
        if transform_args.scale_jitter is not None:
            if (
                transform_args.scale_jitter.min_scale is None
                or transform_args.scale_jitter.max_scale is None
            ):
                scale_range = None
            else:
                scale_range = (
                    transform_args.scale_jitter.min_scale,
                    transform_args.scale_jitter.max_scale,
                )
            self.scale_jitter = Compose(
                [
                    ScaleJitter(
                        sizes=transform_args.scale_jitter.sizes,
                        target_size=(
                            no_auto(transform_args.image_size)
                            if transform_args.scale_jitter.sizes is None
                            else None
                        ),
                        scale_range=scale_range,
                        num_scales=transform_args.scale_jitter.num_scales,
                        divisible_by=transform_args.scale_jitter.divisible_by,
                        p=transform_args.scale_jitter.prob,
                        step_seeding=transform_args.scale_jitter.step_seeding,
                        seed_offset=transform_args.scale_jitter.seed_offset,
                    )
                ],
                bbox_params=transform_args.bbox_params,
            )
        else:
            self.scale_jitter = None

    def __call__(self, batch: list[ObjectDetectionDatasetItem]) -> ObjectDetectionBatch:
        if self.scale_jitter is not None:
            # Turn into numpy again.
            batch_np = [
                {
                    "image_path": item["image_path"],
                    "image": item["image"].permute(1, 2, 0).numpy(),
                    "bboxes": item["bboxes"].numpy(),
                    "classes": item["classes"].numpy(),
                    "original_size": item["original_size"],
                }
                for item in batch
            ]

            # Apply transform.
            seed = np.random.randint(0, 1_000_000)
            images = []
            bboxes = []
            classes = []
            for item in batch_np:
                self.scale_jitter.step = seed
                out = self.scale_jitter(
                    image=item["image"],
                    bboxes=item["bboxes"],
                    class_labels=item["classes"],
                )
                images.append(out["image"])
                bboxes.append(out["bboxes"])
                classes.append(out["class_labels"])

            # Old versions of albumentations return classes/boxes as a list.
            bboxes = [
                bbox if isinstance(bbox, np.ndarray) else np.array(bbox)
                for bbox in bboxes
            ]
            classes = [
                cls_ if isinstance(cls_, np.ndarray) else np.array(cls_)
                for cls_ in classes
            ]

            # Turn back into torch tensors.
            images = [
                torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)
                for img in images
            ]
            bboxes = [torch.from_numpy(bbox).to(torch.float32) for bbox in bboxes]
            classes = [torch.from_numpy(cls).to(torch.int64) for cls in classes]

            out_: ObjectDetectionBatch = {
                "image_path": [item["image_path"] for item in batch],
                "image": torch.stack(images),
                "bboxes": bboxes,
                "classes": classes,
                "original_size": [item["original_size"] for item in batch],
            }
            return out_
        else:
            out_ = {
                "image_path": [item["image_path"] for item in batch],
                "image": torch.stack([item["image"] for item in batch]),
                "bboxes": [item["bboxes"] for item in batch],
                "classes": [item["classes"] for item in batch],
                "original_size": [item["original_size"] for item in batch],
            }
            return out_
