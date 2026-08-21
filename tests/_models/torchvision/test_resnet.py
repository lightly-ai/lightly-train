#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pytest
import torch
from torchvision import models

from lightly_train._models.torchvision.resnet import ResNetModelWrapper


class TestResNetModelWrapper:
    def test_feature_dim(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        assert feature_extractor.feature_dim() == 512

    def test_forward_features(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_features(x)["features"]
        assert features.shape == (1, 512, 7, 7)

    def test_forward_pool(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 512, 7, 7)
        pool = feature_extractor.forward_pool({"features": x})["pooled_features"]
        assert pool.shape == (1, 512, 1, 1)

    def test_get_model(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        assert feature_extractor.get_model() is model

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = models.resnet18()
        wrapped_model = ResNetModelWrapper(model=model)
        wrapped_model.to("meta")
        wrapped_model.forward_features(torch.rand(1, 3, 224, 224, device="meta"))

    def test_multiscale_feature_dims(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        assert feature_extractor.multiscale_feature_dims() == [64, 128, 256, 512]

    def test_multiscale_feature_strides(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        assert feature_extractor.multiscale_feature_strides() == [4, 8, 16, 32]

    def test_forward_multiscale_features(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_multiscale_features(
            x, layer_indices=[0, 1, 2, 3]
        )
        shapes = [feature["features"].shape for feature in features]
        assert shapes == [
            (1, 64, 56, 56),
            (1, 128, 28, 28),
            (1, 256, 14, 14),
            (1, 512, 7, 7),
        ]

    def test_forward_multiscale_features__order(self) -> None:
        # Features are returned in the same order as the requested indices.
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_multiscale_features(
            x, layer_indices=[3, 1]
        )
        shapes = [feature["features"].shape for feature in features]
        assert shapes == [(1, 512, 7, 7), (1, 128, 28, 28)]

    def test_forward_multiscale_features__invalid_index(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with pytest.raises(ValueError):
            feature_extractor.forward_multiscale_features(x, layer_indices=[4])

    def test_forward_multiscale_features__matches_forward_features(self) -> None:
        # The last stage matches forward_features. Eval mode disables stochastic layers
        # so that the two forward passes are deterministic.
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model).eval()
        x = torch.rand(1, 3, 224, 224)
        last = feature_extractor.forward_multiscale_features(x, layer_indices=[3])[0]
        assert torch.equal(
            last["features"], feature_extractor.forward_features(x)["features"]
        )
