#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lightly_train._task_models.picodet_object_detection.config import (
    PICODET_OBJECT_DETECTION_MODEL_REGISTRY,
)


def test_picodet_s_416_config() -> None:
    config = PICODET_OBJECT_DETECTION_MODEL_REGISTRY.get("picodet/s-416")()

    assert config.model_size == "s"
    assert config.image_size == (416, 416)
    assert config.stacked_convs == 2
    assert config.neck_out_channels == 96
    assert config.head_feat_channels == 96


def test_picodet_l_640_config() -> None:
    config = PICODET_OBJECT_DETECTION_MODEL_REGISTRY.get("picodet/l-640")()

    assert config.model_size == "l"
    assert config.image_size == (640, 640)
    assert config.stacked_convs == 3
    assert config.neck_out_channels == 128
    assert config.head_feat_channels == 128


def test_picodet_coco_alias_metadata() -> None:
    aliases = {
        "picodet-s-coco": (
            "picodet_s_coco_416_260303_23022a45.pt",
            "23022a456b2583246288041762a1a66d8d59820d5e775912cb4eb366d3a0cd68",
        ),
        "picodet-l-coco": (
            "picodet_l_coco_640_260303_b1a16990.pt",
            "b1a16990fe4f86fe60aefb2dcb4bf97ead9cc616f6c14ce4638aa2b838351fff",
        ),
    }

    for alias, expected_checkpoint in aliases.items():
        checkpoint = (
            PICODET_OBJECT_DETECTION_MODEL_REGISTRY.get_alias_metadata(
                alias
            ).downloadable_checkpoint
        )
        assert (checkpoint.url, checkpoint.sha256) == expected_checkpoint
