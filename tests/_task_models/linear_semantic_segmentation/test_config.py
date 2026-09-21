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


def test_registry_has_config_for_every_dinov2_backbone() -> None:
    """Every DINOv2 backbone must have an explicit config.

    DINOv2 configs set ``freeze_mask_token=True`` and ``drop_path_rate=0.0``.
    ``Fallback`` keeps ``backbone_name=""``, so ``_check_freeze_mask_token`` does
    not fire for it and an unregistered DINOv2 backbone would silently run with
    ``freeze_mask_token=False``. Backbones of other packages need no extra fields
    and are fine on ``Fallback``.
    """
    missing = [
        model_name
        for model_name in LinearSemanticSegmentation.list_model_names()
        if model_name.startswith("dinov2/")
        and LINEAR_SEG_MODEL_REGISTRY.get(
            model_name, default=LinearSegConfigRegistry.Fallback
        )
        is LinearSegConfigRegistry.Fallback
    ]
    assert not missing, (
        f"Missing explicit LinearSemanticSegmentation configs for DINOv2 backbones: "
        f"{missing}. Add a config for each in LinearSegConfigRegistry."
    )
