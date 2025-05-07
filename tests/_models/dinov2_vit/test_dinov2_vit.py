#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import cast

import torch

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._modules.teachers.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._modules.teachers.dinov2.models.vision_transformer import (
    vit_small as vit_small_untyped,
)


def vit_small() -> DinoVisionTransformer:
    return cast(
        DinoVisionTransformer,
        vit_small_untyped() # type: ignore[no-untyped-call]
    )

class TestDINOv2ViTModelWrapper:
    def test_init(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        for name, param in feature_extractor.named_parameters():
            assert param.requires_grad, name

        for name, module in feature_extractor.named_modules():
            assert module.training, name
    
    def test_feature_dim(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        assert feature_extractor.feature_dim() == 384
    
    def test_forward_features(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_features(x)["features"]
        cls_token = feature_extractor.forward_features(x)["cls_token"]
        assert features.shape == (1, 384, 14, 14)
        assert cls_token.shape == (1, 384)
    
    def test_forward_pool(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        x = torch.rand(1, 384, 14, 14)
        pooled_features = feature_extractor.forward_pool({"features": x})["pooled_features"]
        assert pooled_features.shape == (1, 384, 1, 1)