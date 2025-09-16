#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import (
    Literal,
    Type,
    TypeVar,
)

import pydantic
from lightly.transforms.utils import IMAGENET_NORMALIZE
from pydantic import Field

from lightly_train._configs.config import PydanticConfig
from lightly_train._configs.validate import no_auto
from lightly_train.types import TransformInput, TransformOutput

logger = logging.getLogger(__name__)


class ChannelDropArgs(PydanticConfig):
    num_channels_keep: int
    weight_drop: tuple[float, ...] = Field(strict=False)


class ResizeArgs(PydanticConfig):
    height: int
    width: int


class RandomResizeArgs(PydanticConfig):
    min_scale: float = 0.08
    max_scale: float = 1.0

    def as_tuple(self) -> tuple[float, float]:
        return self.min_scale, self.max_scale


class RandomResizedCropArgs(PydanticConfig):
    # don't allow None for .size since it comes from MethodTransformArgs.image_size
    # however .scale comes from MethodTransformArgs.random_resize which may be None
    size: tuple[int, int]
    scale: RandomResizeArgs | None


class RandomFlipArgs(PydanticConfig):
    horizontal_prob: float = 0.5
    vertical_prob: float = 0.0


class RandomPhotometricDistortArgs(PydanticConfig):
    brightness: tuple[float, float] = Field(strict=False, ge=0.0)
    contrast: tuple[float, float] = Field(strict=False, ge=0.0)
    saturation: tuple[float, float] = Field(strict=False, ge=0.0)
    hue: tuple[float, float] = Field(strict=False, ge=-0.5, le=0.5)
    prob: float = Field(ge=0.0, le=1.0)


class RandomRotationArgs(PydanticConfig):
    prob: float
    degrees: int


class RandomZoomOutArgs(PydanticConfig):
    prob: float = Field(ge=0.0, le=1.0)
    fill: float
    side_range: tuple[float, float] = Field(strict=False, ge=1.0)


class ColorJitterArgs(PydanticConfig):
    prob: float  # Probability to apply ColorJitter
    strength: float  # Multiplier for the parameters below
    brightness: float
    contrast: float
    saturation: float
    hue: float


class GaussianBlurArgs(PydanticConfig):
    prob: float
    sigmas: tuple[float, float]
    blur_limit: int | tuple[int, int]

    # Using strict=False does not work here, because we have a Union type.
    @pydantic.field_validator("blur_limit", mode="before")
    def cast_list_to_tuple(cls, value: int | Sequence[int]) -> int | tuple[int, int]:
        if isinstance(value, int):
            return value
        elif (
            isinstance(value, Sequence)
            and (len(value) == 2)
            and all(isinstance(v, int) for v in value)
        ):
            value = tuple(value)
            assert isinstance(value, tuple)
            assert all(isinstance(v, int) for v in value)
            return value
        else:
            raise ValueError("blur_limit must be an int or a tuple of ints")


class SolarizeArgs(PydanticConfig):
    prob: float
    threshold: float


class NormalizeArgs(PydanticConfig):
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    mean: tuple[float, ...] = Field(
        default=(
            IMAGENET_NORMALIZE["mean"][0],
            IMAGENET_NORMALIZE["mean"][1],
            IMAGENET_NORMALIZE["mean"][2],
        ),
        strict=False,
    )
    std: tuple[float, ...] = Field(
        default=(
            IMAGENET_NORMALIZE["std"][0],
            IMAGENET_NORMALIZE["std"][1],
            IMAGENET_NORMALIZE["std"][2],
        ),
        strict=False,
    )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @classmethod
    def from_dict(cls, config: dict[str, list[float]]) -> NormalizeArgs:
        return cls(
            mean=(config["mean"][0], config["mean"][1], config["mean"][2]),
            std=(config["std"][0], config["std"][1], config["std"][2]),
        )


class ScaleJitterArgs(PydanticConfig):
    min_scale: float
    max_scale: float
    num_scales: int
    prob: float


class SmallestMaxSizeArgs(PydanticConfig):
    # Maximum size of the smallest side of the image.
    max_size: int | list[int] | Literal["auto"]
    prob: float

    def resolve_auto(self, height: int, width: int) -> None:
        if self.max_size == "auto":
            self.max_size = min(height, width)


class RandomCropArgs(PydanticConfig):
    height: int | Literal["auto"]
    width: int | Literal["auto"]
    pad_position: str
    pad_if_needed: bool  # Pad if crop size exceeds image size.
    fill: tuple[float, ...] | float  # Padding value for images.
    prob: float  # Probability to apply RandomCrop.

    def resolve_auto(self, height: int, width: int) -> None:
        if self.height == "auto":
            self.height = height
        if self.width == "auto":
            self.width = width


class MethodTransformArgs(PydanticConfig):
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    image_size: tuple[int, int]
    channel_drop: ChannelDropArgs | None
    num_channels: int | Literal["auto"]
    random_resize: RandomResizeArgs | None
    random_flip: RandomFlipArgs | None
    random_rotation: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None
    random_gray_scale: float | None
    normalize: NormalizeArgs
    gaussian_blur: GaussianBlurArgs | None
    solarize: SolarizeArgs | None

    def resolve_auto(self) -> None:
        if self.num_channels == "auto":
            if self.channel_drop is not None:
                self.num_channels = self.channel_drop.num_channels_keep
            else:
                self.num_channels = len(self.normalize.mean)

    def resolve_incompatible(self) -> None:
        # Adjust normalization mean and std to match num_channels.
        if len(self.normalize.mean) != no_auto(self.num_channels):
            logger.debug(
                "Adjusting mean of normalize transform to match num_channels. "
                f"num_channels is {self.num_channels} but "
                f"normalize.mean has length {len(self.normalize.mean)}."
            )
            # Repeat the values until they match num_channels.
            self.normalize.mean = tuple(
                self.normalize.mean[i % len(self.normalize.mean)]
                for i in range(no_auto(self.num_channels))
            )
        if len(self.normalize.std) != no_auto(self.num_channels):
            logger.debug(
                "Adjusting std of normalize transform to match num_channels. "
                f"num_channels is {self.num_channels} but "
                f"normalize.std has length {len(self.normalize.std)}."
            )
            # Repeat the values until they match num_channels.
            self.normalize.std = tuple(
                self.normalize.std[i % len(self.normalize.std)]
                for i in range(no_auto(self.num_channels))
            )

        # Disable transforms if necessary.
        if self.color_jitter is not None and no_auto(self.num_channels) != 3:
            logger.debug(
                "Disabling color jitter transform as it only supports 3-channel "
                f"images but num_channels is {self.num_channels}."
            )
            self.color_jitter = None
        if self.random_gray_scale is not None and no_auto(self.num_channels) != 3:
            logger.debug(
                "Disabling random gray scale transform as it only supports 3-channel "
                f"images but num_channels is {self.num_channels}."
            )
            self.random_gray_scale = None
        if self.solarize is not None and no_auto(self.num_channels) != 3:
            logger.debug(
                "Disabling solarize transform as it only supports 3-channel "
                f"images but num_channels is {self.num_channels}."
            )
            self.solarize = None


_T = TypeVar("_T", covariant=True)


class MethodTransform:
    transform_args: MethodTransformArgs

    def __init__(self, transform_args: MethodTransformArgs):
        self.transform_args = transform_args

    def __call__(self, input: TransformInput) -> TransformOutput:
        raise NotImplementedError

    @staticmethod
    def transform_args_cls() -> Type[MethodTransformArgs]:
        raise NotImplementedError
