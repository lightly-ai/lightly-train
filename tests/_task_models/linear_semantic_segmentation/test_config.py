#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.linear_semantic_segmentation.config import (
    LINEAR_SEG_MODEL_REGISTRY,
    LinearSegConfigRegistry,
)
from lightly_train._task_models.linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)


def test_registry_has_config_for_every_backbone() -> None:
    """Every backbone in the DINOv2/DINOv3 packages must have an explicit config.

    Guards against forgetting to register a config when a new backbone is added
    to the dinov2 or dinov3 packages. Names that fall through to ``Fallback`` are
    derived at runtime and are not considered registered.
    """
    missing = [
        model_name
        for model_name in LinearSemanticSegmentation.list_model_names()
        if LINEAR_SEG_MODEL_REGISTRY.get(
            model_name, default=LinearSegConfigRegistry.Fallback
        )
        is LinearSegConfigRegistry.Fallback
    ]
    assert not missing, (
        f"Missing explicit LinearSemanticSegmentation configs for backbones: "
        f"{missing}. Add a config for each in LinearSegConfigRegistry."
    )
