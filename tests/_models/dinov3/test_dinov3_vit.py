#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    vit_small as vit_small,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper


class TestDINOv3ViTModelWrapper:
    def test_init(self) -> None:
        model = vit_small()
        feature_extractor = DINOv3ViTModelWrapper(model=model)

        for name, param in feature_extractor.named_parameters():
            assert param.requires_grad, name

        for name, module in feature_extractor.named_modules():
            assert module.training, name

    def test_feature_dim(self) -> None:
        model = vit_small()
        feature_extractor = DINOv3ViTModelWrapper(model=model)

        assert feature_extractor.feature_dim() == 384

    def test_forward_features(self) -> None:
        model = vit_small()
        model.init_weights()  # type: ignore[no-untyped-call]
        feature_extractor = DINOv3ViTModelWrapper(model=model)

        x = torch.rand(1, 3, 224, 224)
        collated_masks = torch.rand(1, 14 * 14) > 0.5

        feats_cls = feature_extractor.forward_features(x)
        assert feats_cls["features"].shape == (1, 384, 14, 14)
        assert feats_cls["cls_token"].shape == (1, 384)

        feats_cls_masked = feature_extractor.forward_features(x, masks=collated_masks)
        assert not torch.allclose(
            feats_cls["features"], feats_cls_masked["features"], atol=1e-6
        )
        assert not torch.allclose(
            feats_cls["cls_token"], feats_cls_masked["cls_token"], atol=1e-6
        )

    def test_forward_pool(self) -> None:
        model = vit_small()
        feature_extractor = DINOv3ViTModelWrapper(model=model)

        x = torch.rand(1, 384, 14, 14)
        x_cls = torch.rand(1, 384)
        pooled_features = feature_extractor.forward_pool(
            {"features": x, "cls_token": x_cls}
        )["pooled_features"]
        print(pooled_features.shape)
        assert pooled_features.shape == (1, 384, 1, 1)

    def test_get_model(self) -> None:
        model = vit_small()
        extractor = DINOv3ViTModelWrapper(model=model)
        assert extractor.get_model() is model

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = vit_small()
        wrapped_model = DINOv3ViTModelWrapper(model=model)
        wrapped_model.to("meta")
        wrapped_model.forward_features(torch.rand(1, 3, 224, 224, device="meta"))
