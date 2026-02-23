#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.channel_drop import ChannelDropTV
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.random_rotate_90 import RandomRotate90
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformInput,
    TaskTransformOutput,
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

        if self.stop_step is not None:
            raise NotImplementedError(
                "Stopping certain augmentations after some steps is not implemented yet."
            )
        self.global_step = 0
        self.stop_ops = (
            transform_args.stop_policy.ops if transform_args.stop_policy else set()
        )
        self.past_stop = False

        transforms_list: list[v2.Transform] = []

        if transform_args.channel_drop is not None:
            transforms_list.append(
                ChannelDropTV(
                    num_channels_keep=transform_args.channel_drop.num_channels_keep,
                    weight_drop=transform_args.channel_drop.weight_drop,
                )
            )

        if transform_args.photometric_distort is not None:
            transforms_list.append(
                v2.RandomPhotometricDistort(
                    brightness=transform_args.photometric_distort.brightness,
                    contrast=transform_args.photometric_distort.contrast,
                    saturation=transform_args.photometric_distort.saturation,
                    hue=transform_args.photometric_distort.hue,
                    p=transform_args.photometric_distort.prob,
                )
            )

        if transform_args.random_zoom_out is not None:
            transforms_list.append(
                v2.RandomZoomOut(
                    fill=transform_args.random_zoom_out.fill,
                    side_range=transform_args.random_zoom_out.side_range,
                    p=transform_args.random_zoom_out.prob,
                )
            )

        if transform_args.random_iou_crop is not None:
            raise NotImplementedError(
                "RandomIoUCrop is not implemented yet for OrientedObjectDetectionTransform."
                "torchvision does not support it for now."
            )

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                transforms_list.append(
                    v2.RandomHorizontalFlip(
                        p=transform_args.random_flip.horizontal_prob
                    )
                )
            if transform_args.random_flip.vertical_prob > 0.0:
                transforms_list.append(
                    v2.RandomVerticalFlip(p=transform_args.random_flip.vertical_prob)
                )

        if transform_args.random_rotate_90 is not None:
            transforms_list.append(
                RandomRotate90(p=transform_args.random_rotate_90.prob)
            )

        if transform_args.random_rotate is not None:
            transforms_list.append(
                v2.RandomApply(
                    [
                        v2.RandomRotation(
                            degrees=transform_args.random_rotate.degrees,  # type: ignore[arg-type]
                            interpolation=v2.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=transform_args.random_rotate.prob,
                )
            )

        if transform_args.resize is not None:
            transforms_list.append(
                v2.Resize(
                    size=(
                        no_auto(transform_args.resize.height),
                        no_auto(transform_args.resize.width),
                    ),
                    antialias=True,
                )
            )

        transforms_list.append(v2.ToDtype(torch.float32, scale=True))

        if transform_args.normalize is not None:
            normalize_args = no_auto(transform_args.normalize)
            transforms_list.append(
                v2.Normalize(
                    mean=list(normalize_args.mean),
                    std=list(normalize_args.std),
                )
            )

        self.transform = v2.Compose(transforms_list)

    def __call__(
        self, input: OrientedObjectDetectionTransformInput
    ) -> OrientedObjectDetectionTransformOutput:
        if (
            self.stop_step is not None
            and self.global_step >= self.stop_step
            and not self.past_stop
        ):
            raise NotImplementedError("Stop policy is not implemented yet.")

        assert "bboxes" in input, (
            "Input must contain bboxes for oriented object detection transform."
        )
        assert "class_labels" in input, (
            "Input must contain class_labels for oriented object detection transform."
        )

        image_hwc = input["image"]
        bboxes_np = input["bboxes"]
        class_labels = input["class_labels"]

        h, w = image_hwc.shape[:2]
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        tv_image = Image(torch.from_numpy(image_chw))
        tv_bboxes = BoundingBoxes(  # type: ignore[call-arg]
            torch.from_numpy(bboxes_np),
            format=BoundingBoxFormat.CXCYWHR,
            canvas_size=(h, w),
        )

        transformed_image, transformed_bboxes = self.transform(tv_image, tv_bboxes)

        return {
            "image": transformed_image,
            "bboxes": transformed_bboxes,
            "class_labels": torch.from_numpy(class_labels),
        }
