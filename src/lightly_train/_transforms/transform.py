#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

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
from lightly_train.types import TransformInput, TransformOutput


class ChannelDropArgs(PydanticConfig):
    num_channels_keep: int
    weight_drop: tuple[float, ...] = Field(strict=False)


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


class RandomRotationArgs(PydanticConfig):
    prob: float
    degrees: int


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
    mean: tuple[float, float, float] = Field(
        default=(
            IMAGENET_NORMALIZE["mean"][0],
            IMAGENET_NORMALIZE["mean"][1],
            IMAGENET_NORMALIZE["mean"][2],
        ),
        strict=False,
    )
    std: tuple[float, float, float] = Field(
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
    random_resize: RandomResizeArgs | None
    random_flip: RandomFlipArgs | None
    random_rotation: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None
    random_gray_scale: float | None
    normalize: NormalizeArgs
    gaussian_blur: GaussianBlurArgs | None
    solarize: SolarizeArgs | None


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
