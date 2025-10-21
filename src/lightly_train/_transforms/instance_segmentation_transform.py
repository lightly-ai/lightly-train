#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from albumentations.pytorch.transforms import ToTensorV2
from torch import Tensor
from typing_extensions import Literal

from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train.types import (
    NDArrayBBoxes,
    NDArrayBinaryMasks,
    NDArrayClasses,
    NDArrayImage,
)


class InstanceSegmentationTransformInput(TaskTransformInput):
    image: NDArrayImage
    binary_masks: NDArrayBinaryMasks
    bboxes: NDArrayBBoxes
    class_labels: NDArrayClasses


class InstanceSegmentationTransformOutput(TaskTransformOutput):
    image: Tensor
    binary_masks: NDArrayBinaryMasks
    bboxes: NDArrayBBoxes
    class_labels: NDArrayClasses


class InstanceSegmentationTransformArgs(TaskTransformArgs):
    num_channels: int | Literal["auto"]


class InstanceSegmentationTransform(TaskTransform):
    transform_args_cls: type[InstanceSegmentationTransformArgs] = (
        InstanceSegmentationTransformArgs
    )

    def __init__(
        self,
        transform_args: InstanceSegmentationTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)
        self.transform = ToTensorV2()

    def __call__(
        self, input: InstanceSegmentationTransformInput
    ) -> InstanceSegmentationTransformOutput:
        transformed: InstanceSegmentationTransformOutput = self.transform(
            image=input["image"],
            binary_masks=input["binary_masks"],
            bboxes=input["bboxes"],
            class_labels=input["class_labels"],
        )
        return transformed
