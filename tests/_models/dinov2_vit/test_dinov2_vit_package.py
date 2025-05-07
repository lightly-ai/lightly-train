#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import cast

import pytest

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
from lightly_train._modules.teachers.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._modules.teachers.dinov2.models.vision_transformer import (
    vit_small as vit_small_untyped,
)

from ...helpers import DummyCustomModel


def vit_small() -> DinoVisionTransformer:
    return cast(
        DinoVisionTransformer,
        vit_small_untyped() # type: ignore[no-untyped-call]
    )

class TestDINOv2ViTPackage:
    @pytest.mark.parametrize(
        "model_name, supported",
        [
            ("dinov2_vit/dinov2_vits14", False),
            ("dinov2_vit/dinov2_vitb14", False),
            ("dinov2_vit/dinov2_vitl14", False),
            ("dinov2_vit/dinov2_vitg14", False),
            ("dinov2_vit/vits14", True),
            ("dinov2_vit/vitb14", True),
            ("dinov2_vit/vitl14", True),
            ("dinov2_vit/vitg14", True),
        ],
    )
    def test_list_model_names(self, model_name: str, supported: bool) -> None:
        model_names = DINOv2ViTPackage.list_model_names()
        assert (model_name in model_names) is supported
    
    def test_is_supported_model__true(self) -> None:
        model = vit_small()
        assert DINOv2ViTPackage.is_supported_model(model)

    def test_is_supported_model__false(self) -> None:
        model = DummyCustomModel()
        assert not DINOv2ViTPackage.is_supported_model(model)
    
    @pytest.mark.parametrize(
        "model_name",
        ["vits14", "vitb14"],
    )
    def test_get_model(self, model_name: str) -> None:
        model = DINOv2ViTPackage.get_model(model_name=model_name)
        assert isinstance(model, DinoVisionTransformer)

    def test_get_model_wrapper(self) -> None:
        model = vit_small()
        fe = DINOv2ViTPackage.get_model_wrapper(model=model)
        assert isinstance(fe, DINOv2ViTModelWrapper)
    
    #TODO test export