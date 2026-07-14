#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import (
    DownloadableCheckpoint,
    ModelAlias,
    ModelRegistry,
)


class PicoDetObjectDetectionConfig(PydanticConfig):
    model_size: Literal["s", "m", "l"]
    image_size: tuple[int, int]
    stacked_convs: int
    neck_out_channels: int
    head_feat_channels: int


PICODET_OBJECT_DETECTION_MODEL_REGISTRY: ModelRegistry[PicoDetObjectDetectionConfig] = (
    ModelRegistry()
)

_PICODET_S_COCO_URL = "picodet_s_coco_416_260303_23022a45.pt"
_PICODET_S_COCO_SHA256 = (
    "23022a456b2583246288041762a1a66d8d59820d5e775912cb4eb366d3a0cd68"
)
_PICODET_L_COCO_URL = "picodet_l_coco_640_260303_b1a16990.pt"
_PICODET_L_COCO_SHA256 = (
    "b1a16990fe4f86fe60aefb2dcb4bf97ead9cc616f6c14ce4638aa2b838351fff"
)


class PicoDetObjectDetectionConfigRegistry(ConfigsNamespace):
    @PICODET_OBJECT_DETECTION_MODEL_REGISTRY.register(
        ModelAlias(
            name="picodet-s-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_PICODET_S_COCO_URL,
                sha256=_PICODET_S_COCO_SHA256,
            ),
        ),
        "picodet/s-416",
    )
    class Small416(PicoDetObjectDetectionConfig):
        model_size: Literal["s", "m", "l"] = "s"
        image_size: tuple[int, int] = (416, 416)
        stacked_convs: int = 2
        neck_out_channels: int = 96
        head_feat_channels: int = 96

    @PICODET_OBJECT_DETECTION_MODEL_REGISTRY.register(
        ModelAlias(
            name="picodet-l-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_PICODET_L_COCO_URL,
                sha256=_PICODET_L_COCO_SHA256,
            ),
        ),
        "picodet/l-640",
    )
    class Large640(PicoDetObjectDetectionConfig):
        model_size: Literal["s", "m", "l"] = "l"
        image_size: tuple[int, int] = (640, 640)
        stacked_convs: int = 3
        neck_out_channels: int = 128
        head_feat_channels: int = 128
