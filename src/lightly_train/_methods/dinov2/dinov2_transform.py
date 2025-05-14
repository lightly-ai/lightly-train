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


class DINOv2RandomResizeArgs(DINORandomResizeArgs):
    min_scale: float = 0.32


class DINOv2LocalViewRandomResizeArgs(DINOLocalViewRandomResizeArgs):
    max_scale: float = 0.32


class DINOv2ViTSBLocalViewTransformArgs(DINOLocalViewTransformArgs):
    num_views: int = 8
    random_resize: DINOv2LocalViewRandomResizeArgs | None = Field(
        default_factory=DINOv2LocalViewRandomResizeArgs
    )


class DINOv2ViTLGLocalViewTransformArgs(DINOLocalViewTransformArgs):
    num_views: int = 8
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    view_size: tuple[int, int] = Field(default=(98, 98), strict=False)
    random_resize: DINOv2LocalViewRandomResizeArgs | None = Field(
        default_factory=DINOv2LocalViewRandomResizeArgs
    )


class DINOv2ViTSBTransformArgs(DINOTransformArgs):
    random_resize: DINOv2RandomResizeArgs | None = Field(
        default_factory=DINOv2RandomResizeArgs
    )
    local_view: DINOv2ViTSBLocalViewTransformArgs | None = Field(
        default_factory=DINOv2ViTSBLocalViewTransformArgs
    )


class DINOv2ViTLGTransformArgs(DINOTransformArgs):
    random_resize: DINOv2RandomResizeArgs | None = Field(
        default_factory=DINOv2RandomResizeArgs
    )
    local_view: DINOv2ViTLGLocalViewTransformArgs | None = Field(
        default_factory=DINOv2ViTLGLocalViewTransformArgs
    )


class DINOv2ViTSBTransform(DINOTransform):
    @staticmethod
    def transform_args_cls() -> type[DINOv2ViTSBTransformArgs]:
        return DINOv2ViTSBTransformArgs


class DINOv2ViTLGTransform(DINOTransform):
    @staticmethod
    def transform_args_cls() -> type[DINOv2ViTLGTransformArgs]:
        return DINOv2ViTLGTransformArgs