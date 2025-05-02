#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from importlib import util as importlib_util

import pytest
import torch

from lightly_train._models.rfdetr.rfdetr import RFDETRFeatureExtractor

if importlib_util.find_spec("rfdetr") is None:
    pytest.skip("rfdetr is not installed", allow_module_level=True)

from rfdetr.detr import RFDETRBase


class TestRFDETRFeatureExtractor:
    def test_init(self) -> None:
        model = RFDETRBase().model.model  # type: ignore[no-untyped-call]
        feature_extractor = RFDETRFeatureExtractor(model=model)

        for name, param in feature_extractor.named_parameters():
            assert param.requires_grad, name

        for name, module in feature_extractor.named_modules():
            assert module.training, name

    def test_feature_dim(self) -> None:
        model = RFDETRBase().model.model  # type: ignore[no-untyped-call]

        feature_extractor = RFDETRFeatureExtractor(model=model)

        assert feature_extractor.feature_dim() == 384

    def test_forward_features(
        self,
    ) -> None:
        model_instance = RFDETRBase()  # type: ignore[no-untyped-call]
        model = model_instance.model.model
        device = model_instance.model.device

        feature_extractor = RFDETRFeatureExtractor(model=model)

        image_size = 224
        expected_dim = feature_extractor.feature_dim()
        x = torch.rand(1, 3, image_size, image_size).to(device=device)
        features = feature_extractor.forward_features(x)["features"]

        assert features.shape == (
            1,
            expected_dim,
            int(image_size // 14),  # we use vit-14 as the backbone
            int(image_size // 14),
        )

    def test_forward_pool(self) -> None:
        model_instance = RFDETRBase()  # type: ignore[no-untyped-call]
        model = model_instance.model.model
        device = model_instance.model.device

        feature_extractor = RFDETRFeatureExtractor(model=model)

        expected_dim = feature_extractor.feature_dim()
        x = torch.rand(1, expected_dim, 7, 7).to(device=device)
        pool = feature_extractor.forward_pool({"features": x})["pooled_features"]

        assert pool.shape == (1, expected_dim, 1, 1)

    def test_get_model(self) -> None:
        model_instance = RFDETRBase()
        model = model_instance.model.model
        feature_extractor = RFDETRFeatureExtractor(model=model)
        model = feature_extractor.get_model()
        assert model is model_instance.model.model
