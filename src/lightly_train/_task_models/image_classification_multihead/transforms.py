#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from lightly_train._transforms.image_classification_transform import (
    ImageClassificationTransform,
    ImageClassificationTransformArgs,
)
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomRotate90Args,
    RandomRotationArgs,
)
from lightly_train.types import ImageSizeTuple


class ImageClassificationRandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.08
    max_scale: float = 1.0


class ImageClassificationMultiheadTrainTransformArgs(ImageClassificationTransformArgs):
    """
    Defines default transform arguments for image classification multihead training.
    """

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_crop: ImageClassificationRandomResizeArgs | None = Field(
        default_factory=ImageClassificationRandomResizeArgs
    )
    resize_scale: float | None = None
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    color_jitter: ColorJitterArgs | None = None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)
        if self.image_size == "auto":
            self.image_size = tuple(model_init_args.get("image_size", (224, 224)))  # type: ignore[assignment]

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize")
            if normalize is None:
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = len(self.normalize.mean)


class ImageClassificationMultiheadValTransformArgs(ImageClassificationTransformArgs):
    """
    Defines default transform arguments for image classification multihead validation.
    """

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    channel_drop: ChannelDropArgs | None = None
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_crop: ImageClassificationRandomResizeArgs | None = None
    resize_scale: float | None = 1.143
    random_flip: RandomFlipArgs | None = None
    random_rotate_90: RandomRotate90Args | None = None
    random_rotate: RandomRotationArgs | None = None
    color_jitter: ColorJitterArgs | None = None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        super().resolve_auto(model_init_args=model_init_args)
        if self.image_size == "auto":
            self.image_size = tuple(model_init_args.get("image_size", (224, 224)))  # type: ignore[assignment]

        height, width = self.image_size
        for field_name in self.__class__.model_fields:
            field = getattr(self, field_name)
            if hasattr(field, "resolve_auto"):
                field.resolve_auto(height=height, width=width)

        if self.normalize == "auto":
            normalize = model_init_args.get("image_normalize")
            if normalize is None:
                self.normalize = NormalizeArgs()
            else:
                assert isinstance(normalize, dict)
                self.normalize = NormalizeArgs.from_dict(normalize)

        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = len(self.normalize.mean)


class ImageClassificationMultiheadTrainTransform(ImageClassificationTransform):
    transform_args_cls = ImageClassificationMultiheadTrainTransformArgs


class ImageClassificationMultiheadValTransform(ImageClassificationTransform):
    transform_args_cls = ImageClassificationMultiheadValTransformArgs
