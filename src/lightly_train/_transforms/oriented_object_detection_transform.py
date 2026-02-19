#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
from albumentations import (
    Compose,
    ToFloat,
)
from albumentations.pytorch.transforms import ToTensorV2
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.random_photometric_distort import (
    RandomPhotometricDistort,
)
from lightly_train._transforms.random_zoom_out import RandomZoomOut
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train._transforms.torchvision_dispatcher import (
    TorchVisionHorizontalFlip,
    TorchVisionRandomIoUCrop,
    TorchVisionResize,
    TorchVisionRotate,
    TorchVisionRotate90,
    TorchVisionVerticalFlip,
)
from lightly_train.types import NDArrayImage, NDArrayOBBoxes


class OrientedObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NotRequired[NDArrayOBBoxes]
    class_labels: NotRequired[NDArray[np.int64]]


class OrientedObjectDetectionTransformOutput(TaskTransformOutput):
    image: Tensor
    bboxes: NotRequired[Tensor]
    class_labels: NotRequired[Tensor]


class OrientedObjectDetectionTransformArgs(ObjectDetectionTransformArgs):
    pass


class OrientedObjectDetectionTransform(TaskTransform):
    transform_args_cls: type[OrientedObjectDetectionTransformArgs] = (
        OrientedObjectDetectionTransformArgs
    )

    def __init__(
        self,
        transform_args: OrientedObjectDetectionTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform_args: OrientedObjectDetectionTransformArgs = transform_args
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
                TorchVisionRandomIoUCrop(
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
                    TorchVisionHorizontalFlip(
                        p=transform_args.random_flip.horizontal_prob
                    )
                ]
            if transform_args.random_flip.vertical_prob > 0.0:
                self.individual_transforms += [
                    TorchVisionVerticalFlip(p=transform_args.random_flip.vertical_prob)
                ]

        if transform_args.random_rotate_90 is not None:
            self.individual_transforms += [
                TorchVisionRotate90(p=transform_args.random_rotate_90.prob)
            ]
        if transform_args.random_rotate is not None:
            self.individual_transforms += [
                TorchVisionRotate(
                    degrees=transform_args.random_rotate.degrees,
                    interpolation=transform_args.random_rotate.interpolation,
                    p=transform_args.random_rotate.prob,
                )
            ]

        if transform_args.resize is not None:
            self.individual_transforms += [
                TorchVisionResize(
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

        self.individual_transforms += [
            ToTensorV2(),
        ]

        self.transform = Compose(
            self.individual_transforms,
            bbox_params=transform_args.bbox_params,
        )

    def __call__(
        self, input: OrientedObjectDetectionTransformInput
    ) -> OrientedObjectDetectionTransformOutput:
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

        return {
            "image": transformed["image"],
            "bboxes": transformed["bboxes"],
            "class_labels": transformed["class_labels"],
        }
