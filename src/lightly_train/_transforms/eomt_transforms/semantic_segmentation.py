#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

import torch
from albumentations import Compose
from torch import Tensor
from typing_extensions import NotRequired

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
    MaskSemanticSegmentationBatch,
    MaskSemanticSegmentationDatasetItem,
    NDArrayImage,
    NDArrayMask,
)


class EoMTSemanticSegmentationTransformInput(TaskTransformInput):
    image: NDArrayImage
    mask: NotRequired[NDArrayMask]


class EoMTSemanticSegmentationTransformOutput(TaskTransformOutput):
    image: Tensor
    mask: NotRequired[Tensor]


class EoMTSemanticSegmentationTransformArgs(TaskTransformArgs):
    ignore_index: int
    image_size: ImageSizeTuple | Literal["auto"]
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

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        pass

    def resolve_incompatible(self) -> None:
        _resolve_incompatible_eomt(transform_args=self)


class EoMTSemanticSegmentationTransform(TaskTransform):
    transform_args_cls: type[EoMTSemanticSegmentationTransformArgs] = (
        EoMTSemanticSegmentationTransformArgs
    )

    def __init__(
        self,
        transform_args: EoMTSemanticSegmentationTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform = Compose(
            _build_transforms_eomt(
                transform_args=transform_args,
                random_crop_fill_mask=transform_args.ignore_index,
            ),
            additional_targets={"mask": "mask"},
        )

    def __call__(
        self, input: EoMTSemanticSegmentationTransformInput
    ) -> EoMTSemanticSegmentationTransformOutput:
        transformed = self.transform(image=input["image"], mask=input["mask"])
        return {"image": transformed["image"], "mask": transformed["mask"]}


class EoMTSemanticSegmentationCollateFunction(TaskCollateFunction):
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
