#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._models import package_helpers
from lightly_train._models.package import MultiScaleFeaturePackage
from lightly_train._task_models.dinov2_linear_semantic_segmentation.config import (
    LINEAR_SEG_MODEL_REGISTRY,
    LinearSegConfigRegistry,
    LinearSemanticSegmentationConfig,
)
from lightly_train._task_models.dinov2_linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)


def _backbone_names() -> list[str]:
    """All backbone names exposed by MultiScaleFeaturePackages."""
    return [
        name
        for pkg in package_helpers.list_packages()
        if isinstance(pkg, MultiScaleFeaturePackage)
        for name in pkg.list_model_names()
    ]


def _registered_config(model_name: str) -> type[LinearSemanticSegmentationConfig]:
    return LINEAR_SEG_MODEL_REGISTRY.get(
        model_name, default=LinearSegConfigRegistry.Fallback
    )


class TestLinearSegConfigRegistry:
    @pytest.mark.parametrize(
        "model_name", LinearSemanticSegmentation.list_model_names()
    )
    def test_registry_in_sync__every_backbone_is_registered(
        self, model_name: str
    ) -> None:
        """Every backbone-derived model name must have an explicit config.

        Names that fall through to ``Fallback`` are derived at runtime and, for
        DINOv2 backbones, leave ``freeze_mask_token=False`` -- which reintroduces
        DDP unused-parameter errors during full fine-tuning (the mask token is
        unused in the segmentation forward pass). Registering each backbone
        explicitly keeps the freeze behaviour under our control. If this fails, add
        a config for the new backbone in ``LinearSegConfigRegistry`` instead of
        relying on ``Fallback``.
        """
        config_cls = _registered_config(model_name)
        assert config_cls is not LinearSegConfigRegistry.Fallback, (
            f"Model '{model_name}' is not registered and falls through to Fallback. "
            f"Add an explicit config for it in LinearSegConfigRegistry."
        )

    @pytest.mark.parametrize("backbone_name", _backbone_names())
    def test_registry_in_sync__dinov2_freezes_mask_token(
        self, backbone_name: str
    ) -> None:
        """DINOv2 backbones must freeze the (unused) mask token for DDP fine-tuning.

        DINOv2's forward pass leaves ``mask_token`` out of the graph when no masks
        are provided, so it stays unused and breaks DDP without
        ``find_unused_parameters=True``. DINOv3 keeps it in the graph via
        ``cls_token + 0 * mask_token``, so it does not need (and must not set, per
        the config validator) ``freeze_mask_token``.
        """
        model_name = f"{backbone_name}-{LinearSemanticSegmentation.model_suffix}"
        config = _registered_config(model_name)()
        if backbone_name.startswith("dinov2/"):
            assert config.freeze_mask_token, (
                f"DINOv2 backbone '{backbone_name}' must set freeze_mask_token=True "
                f"so the unused mask token does not break DDP during full "
                f"fine-tuning."
            )
        else:
            assert not config.freeze_mask_token, (
                f"Backbone '{backbone_name}' is not DINOv2 and must not set "
                f"freeze_mask_token (the config validator forbids it)."
            )

    def test_freeze_mask_token__rejects_non_dinov2_backbone(self) -> None:
        with pytest.raises(ValueError, match="only supported for DINOv2"):
            LinearSemanticSegmentationConfig(
                backbone_name="dinov3/vits16", freeze_mask_token=True
            )

    def test_fallback__derives_backbone_and_does_not_freeze(self) -> None:
        config = LinearSegConfigRegistry.Fallback()
        assert config.backbone_name == ""
        assert config.freeze_mask_token is False
