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
from albumentations import BboxParams, Compose, HorizontalFlip, VerticalFlip
from albumentations.pytorch.transforms import ToTensorV2
from numpy.typing import NDArray
from pydantic import ConfigDict
from torch import Tensor
from typing_extensions import NotRequired

from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.random_photometric_distort import (
    RandomPhotometricDistort,
)
from lightly_train._transforms.random_zoom_out import RandomZoomOut
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ScaleJitterArgs,
    StopPolicyArgs,
)
from lightly_train.types import NDArrayImage


class ObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NotRequired[NDArray[np.float64]]
    class_labels: NotRequired[NDArray[np.int64]]


class ObjectDetectionTransformOutput(TaskTransformOutput):
    image: Tensor
    bboxes: NotRequired[Tensor]
    class_labels: NotRequired[Tensor]


class ObjectDetectionTransformArgs(TaskTransformArgs):
    channel_drop: ChannelDropArgs | None
    num_channels: int | Literal["auto"]
    photometric_distort: RandomPhotometricDistortArgs | None
    random_zoom_out: RandomZoomOutArgs | None
    # TODO: Lionel (09/25): Add RandomIoUCrop
    random_flip: RandomFlipArgs | None
    image_size: tuple[int, int]
    # TODO: Lionel (09/25): Add Normalize
    stop_policy: StopPolicyArgs | None
    scale_jitter: ScaleJitterArgs | None
    bbox_params: BboxParams | None

    # Necessary for the StopPolicyArgs, which are not serializable by pydantic.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def resolve_auto(self) -> None:
        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                # TODO: Lionel (09/25): Get num_channels from normalization.
                self.num_channels = 3

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

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

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                self.individual_transforms += [
                    HorizontalFlip(p=transform_args.random_flip.horizontal_prob)
                ]
            if transform_args.random_flip.vertical_prob > 0.0:
                self.individual_transforms += [
                    VerticalFlip(p=transform_args.random_flip.vertical_prob)
                ]

        self.individual_transforms += [
            ToTensorV2(),
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

        # TODO: Lionel (09/25): Remove in favor of Normalize transform.
        transformed["image"] = transformed["image"] / 255.0

        return {
            "image": transformed["image"],
            "bboxes": transformed["bboxes"],
            "class_labels": transformed["class_labels"],
        }
