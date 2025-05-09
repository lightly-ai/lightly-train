#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torchvision import models

from lightly_train._models.torchvision.convnext import ConvNeXtFeatureExtractor


class TestConvNeXtFeatureExtractor:
    def test_feature_dim(self) -> None:
        model = models.convnext_tiny()
        feature_extractor = ConvNeXtFeatureExtractor(model=model)
        assert feature_extractor.feature_dim() == 768

    def test_forward_features(self) -> None:
        model = models.convnext_tiny()
        feature_extractor = ConvNeXtFeatureExtractor(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_features(x)["features"]
        assert features.shape == (1, 768, 7, 7)

    def test_forward_pool(self) -> None:
        model = models.convnext_tiny()
        feature_extractor = ConvNeXtFeatureExtractor(model=model)
        x = torch.rand(1, 768, 7, 7)
        pool = feature_extractor.forward_pool({"features": x})["pooled_features"]
        assert pool.shape == (1, 768, 1, 1)

    def test_get_model(self) -> None:
        model = models.convnext_tiny()
        feature_extractor = ConvNeXtFeatureExtractor(model=model)
        assert feature_extractor.get_model() is model
