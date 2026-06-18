#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.dinov2_linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)


class TestLinearSemanticSegmentation:
    def test_init__freezes_dinov2_mask_token(self) -> None:
        """A registered DINOv2 model disables grads on the unused mask token.

        The segmentation head never consumes the mask token, so leaving its grad
        enabled breaks DDP during full fine-tuning (backbone_freeze=False). This
        checks the config flag is wired through to the actual parameter.
        """
        model = LinearSemanticSegmentation(
            model_name="dinov2/vits14-linear",
            classes={0: "background", 1: "car"},
            class_ignore_index=None,
            backbone_freeze=False,
            image_size=(14, 14),
            image_normalize={
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.224, 0.225),
            },
            load_weights=False,
        )
        assert model.backbone.get_model().mask_token.requires_grad is False
