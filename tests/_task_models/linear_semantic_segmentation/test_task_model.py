#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch

from lightly_train._task_models.linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)

_IMAGE_NORMALIZE = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
_CLASSES = {0: "background", 1: "car"}


def _make_model(model_name: str, image_size: tuple[int, int]) -> LinearSemanticSegmentation:
    return LinearSemanticSegmentation(
        model_name=model_name,
        classes=_CLASSES,
        class_ignore_index=None,
        backbone_freeze=False,
        image_size=image_size,
        image_normalize=_IMAGE_NORMALIZE,
        load_weights=False,
    )


# --- is_supported_model ---


def test_is_supported_model__dinov2() -> None:
    assert LinearSemanticSegmentation.is_supported_model("dinov2/vits14-linear")


def test_is_supported_model__dinov3_vit() -> None:
    assert LinearSemanticSegmentation.is_supported_model("dinov3/vitt16-linear")


def test_is_supported_model__dinov3_convnext() -> None:
    assert LinearSemanticSegmentation.is_supported_model("dinov3/convnext-tiny-linear")


def test_is_supported_model__fallback_dinov2_arbitrary() -> None:
    # Fallback path: not in registry but structurally valid.
    assert LinearSemanticSegmentation.is_supported_model("dinov2/_vittest14-linear")


def test_is_supported_model__fallback_dinov3_arbitrary() -> None:
    assert LinearSemanticSegmentation.is_supported_model("dinov3/_vittest16-linear")


def test_is_supported_model__unsupported() -> None:
    assert not LinearSemanticSegmentation.is_supported_model("dinov2/vits14")
    assert not LinearSemanticSegmentation.is_supported_model("unknown/model-linear")


# --- forward pass with different backbone types ---


@pytest.fixture()
def dinov2_model() -> LinearSemanticSegmentation:
    return _make_model("dinov2/_vittest14-linear", image_size=(14, 14))


@pytest.fixture()
def dinov3_vit_model() -> LinearSemanticSegmentation:
    return _make_model("dinov3/_vittest16-linear", image_size=(16, 16))


@pytest.fixture()
def dinov3_convnext_model() -> LinearSemanticSegmentation:
    return _make_model("dinov3/_convnexttest-linear", image_size=(32, 32))


def test_forward_train__dinov2(dinov2_model: LinearSemanticSegmentation) -> None:
    x = torch.rand(2, 3, 14, 14)
    logits = dinov2_model.forward_train(x)
    assert logits.shape[:2] == (2, len(_CLASSES))
    assert logits.shape[2:] == (14, 14)


def test_forward_train__dinov3_vit(dinov3_vit_model: LinearSemanticSegmentation) -> None:
    x = torch.rand(2, 3, 16, 16)
    logits = dinov3_vit_model.forward_train(x)
    assert logits.shape[:2] == (2, len(_CLASSES))
    assert logits.shape[2:] == (16, 16)


def test_forward_train__dinov3_convnext(
    dinov3_convnext_model: LinearSemanticSegmentation,
) -> None:
    x = torch.rand(2, 3, 32, 32)
    logits = dinov3_convnext_model.forward_train(x)
    assert logits.shape[:2] == (2, len(_CLASSES))
    assert logits.shape[2:] == (32, 32)


def test_predict_batch__composes_stages_in_order(
    dinov3_vit_model: LinearSemanticSegmentation,
) -> None:
    images = [torch.rand(3, 24, 32), torch.rand(3, 32, 24)]
    result = dinov3_vit_model.predict_batch(images=images)
    assert len(result) == 2
    for mask, img in zip(result, images):
        assert mask.shape == img.shape[1:]


# --- config guard ---


def test_freeze_mask_token__raises_for_non_dinov2() -> None:
    from lightly_train._task_models.linear_semantic_segmentation.config import (
        LinearSemanticSegmentationConfig,
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="freeze_mask_token"):
        LinearSemanticSegmentationConfig(
            backbone_name="dinov3/vitt16", freeze_mask_token=True
        )
