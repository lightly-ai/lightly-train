#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from pydantic import Field

from lightly_train._methods.dino.dino_transform import (
    DINOLocalViewRandomResizeArgs,
    DINOLocalViewTransformArgs,
    DINORandomResizeArgs,
    DINOTransform,
    DINOTransformArgs,
)
from lightly_train.types import ImageSizeTuple


class DINOv2RandomResizeArgs(DINORandomResizeArgs):
    min_scale: float = 0.32


class DINOv2LocalViewRandomResizeArgs(DINOLocalViewRandomResizeArgs):
    max_scale: float = 0.32


class DINOv2ViTLocalViewTransformArgs(DINOLocalViewTransformArgs):
    num_views: int = 8
    view_size: ImageSizeTuple = (98, 98)
    random_resize: DINOv2LocalViewRandomResizeArgs | None = Field(
        default_factory=DINOv2LocalViewRandomResizeArgs
    )


class DINOv2ViTTransformArgs(DINOTransformArgs):
    random_resize: DINOv2RandomResizeArgs | None = Field(
        default_factory=DINOv2RandomResizeArgs
    )
    local_view: DINOv2ViTLocalViewTransformArgs | None = Field(
        default_factory=DINOv2ViTLocalViewTransformArgs
    )


class DINOv2ViTTransform(DINOTransform):
    @staticmethod
    def transform_args_cls() -> type[DINOv2ViTTransformArgs]:
        return DINOv2ViTTransformArgs
