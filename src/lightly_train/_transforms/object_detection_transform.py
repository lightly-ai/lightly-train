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
            self.past_stop = True

        transformed = self.transform(
            image=input["image"],
            bboxes=input["bboxes"],
            class_labels=input["class_labels"],
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
        self.scale_jitter: BatchReplayCompose | None = None

        if transform_args.scale_jitter is not None:
            self.scale_jitter = BatchReplayCompose(
                transforms=[
                    ScaleJitter(
                        sizes=transform_args.scale_jitter.sizes,
                        target_size=(
                            no_auto(transform_args.image_size)
                            if transform_args.scale_jitter.sizes is None
                            else None
                        ),
                        scale_range=transform_args.scale_jitter.scale_range,
                        num_scales=transform_args.scale_jitter.num_scales,
                        divisible_by=transform_args.scale_jitter.divisible_by,
                        p=transform_args.scale_jitter.prob,
                    )
                ],
                bbox_params=transform_args.bbox_params,
            )

        self.to_tensor = BatchTransform(
            Compose(
                transforms=[ToTensorV2()],
                bbox_params=transform_args.bbox_params,
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

        if self.scale_jitter is not None:
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
