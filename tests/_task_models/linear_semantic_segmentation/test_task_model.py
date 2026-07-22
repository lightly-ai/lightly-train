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

_IMAGE_NORMALIZE: dict[str, tuple[float, ...]] = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
_CLASSES = {0: "background", 1: "car"}


def _make_model(
    model_name: str, image_size: tuple[int, int]
) -> LinearSemanticSegmentation:
    return LinearSemanticSegmentation(
        model_name=model_name,
        classes=_CLASSES,
        class_ignore_index=None,
        backbone_freeze=False,
        image_size=image_size,
        image_normalize=_IMAGE_NORMALIZE,
        load_weights=False,
    )


class TestLinearSemanticSegmentation:
    @pytest.mark.parametrize(
        "model_name",
        [
            "dinov2/vits14-linear",
            "dinov2/vitb14-tipsv2-linear",
            "dinov2/vitl14-tipsv2-linear",
            "dinov2/vitso400m14-tipsv2-linear",
            "dinov2/vitg14-tipsv2-linear",
            "dinov3/vitt16-linear",
            "dinov3/convnext-tiny-linear",
            "dinov2/_vittest14-linear",
            "dinov3/_vittest16-linear",
        ],
    )
    def test_is_supported_model__supported(self, model_name: str) -> None:
        assert LinearSemanticSegmentation.is_supported_model(model_name)

    @pytest.mark.parametrize(
        "model_name",
        [
            "dinov2/vits14",
            "unknown/model-linear",
        ],
    )
    def test_is_supported_model__unsupported(self, model_name: str) -> None:
        assert not LinearSemanticSegmentation.is_supported_model(model_name)

    @pytest.mark.parametrize(
        "model_name,image_size",
        [
            ("dinov2/_vittest14-linear", (14, 14)),
            ("dinov3/_vittest16-linear", (16, 16)),
            ("dinov3/_convnexttest-linear", (32, 32)),
        ],
    )
    def test_forward_train__output_shape(
        self, model_name: str, image_size: tuple[int, int]
    ) -> None:
        model = _make_model(model_name, image_size)
        h, w = image_size
        x = torch.rand(2, 3, h, w)
        logits = model.forward_train(x)
        assert logits.shape[:2] == (2, len(_CLASSES))
        assert logits.shape[2:] == image_size

    def test_predict_batch__composes_stages_in_order(self) -> None:
        model = _make_model("dinov3/_vittest16-linear", image_size=(16, 16))
        images = [torch.rand(3, 24, 32), torch.rand(3, 32, 24)]
        result = model.predict_batch(images=images)
        assert len(result) == 2
        for mask, img in zip(result, images):
            assert mask.shape == img.shape[1:]

    def test_init__freezes_dinov2_mask_token(self) -> None:
        """A registered DINOv2 model disables grads on the unused mask token.

        The segmentation head never consumes the mask token, so leaving its grad
        enabled breaks DDP during full fine-tuning (backbone_freeze=False). This
        checks the config flag is wired through to the actual parameter.
        """
        model = _make_model("dinov2/vits14-linear", image_size=(14, 14))
        assert model.backbone.get_model().mask_token.requires_grad is False

    @pytest.mark.parametrize(
        "model_name",
        [
            "dinov2/vitb14-tipsv2-linear",
            "dinov2/vitl14-tipsv2-linear",
            "dinov2/vitso400m14-tipsv2-linear",
            "dinov2/vitg14-tipsv2-linear",
        ],
    )
    def test_tipsv2_config__freezes_mask_token(self, model_name: str) -> None:
        from lightly_train._task_models.linear_semantic_segmentation.config import (
            LINEAR_SEG_MODEL_REGISTRY,
        )

        config = LINEAR_SEG_MODEL_REGISTRY.get(model_name)()
        assert config.backbone_name == model_name[: -len("-linear")]
        assert config.freeze_mask_token is True


class TestLinearSemanticSegmentationConfig:
    def test_freeze_mask_token__raises_for_non_dinov2(self) -> None:
        from pydantic import ValidationError

        from lightly_train._task_models.linear_semantic_segmentation.config import (
            LinearSemanticSegmentationConfig,
        )

        with pytest.raises(ValidationError, match="freeze_mask_token"):
            LinearSemanticSegmentationConfig(
                backbone_name="dinov3/vitt16", freeze_mask_token=True
            )
