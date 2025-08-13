#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torch import Tensor, nn

from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.task_model import TaskModel


class DINOv2Classification(TaskModel):
    """DINOv2 model for image classification tasks."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        backbone_weights: str | None = None,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__(locals(), ignore_args={"backbone_weights"})

        backbone_args = {"drop_path_rate": drop_path_rate}
        backbone_name = model_name
        self.backbone = DINOV2_VIT_PACKAGE.get_model(
            model_name=backbone_name,
            model_args=backbone_args,
        )
        embed_dim = getattr(self.backbone, "embed_dim", 384)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Load backbone weights if provided
        if backbone_weights is not None:
            state_dict = torch.load(backbone_weights, map_location="cpu")
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor) -> Tensor:
        # Assume backbone returns (B, N+1, D), where first token is class token
        tokens = self.backbone(x)  # (B, N+1, D)
        class_token = tokens[:, 0]  # (B, D)
        logits = self.classifier(class_token)  # (B, num_classes)
        assert isinstance(logits, Tensor)
        return logits
