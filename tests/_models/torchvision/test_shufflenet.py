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

from lightly_train._models.torchvision.shufflenet import ShuffleNetV2ModelWrapper


class TestShuffleNetV2ModelWrapper:
    def test_feature_dim(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        assert wrapped_model.feature_dim() == 1024

    def test_forward_features(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = wrapped_model.forward_features(x)["features"]
        assert features.shape == (1, 1024, 7, 7)

    def test_forward_pool(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        x = torch.rand(1, 1024, 7, 7)
        pool = wrapped_model.forward_pool({"features": x})["pooled_features"]
        assert pool.shape == (1, 1024, 1, 1)

    def test_get_model(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        assert wrapped_model.get_model() is model

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        wrapped_model.to("meta")
        wrapped_model.forward_features(torch.rand(1, 3, 224, 224, device="meta"))

    def test_multiscale_feature_dims__not_supported(self) -> None:
        # Multi-scale feature extraction is not supported for ShuffleNetV2.
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        with pytest.raises(NotImplementedError):
            wrapped_model.multiscale_feature_dims()

    def test_forward_multiscale_features__not_supported(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        with pytest.raises(NotImplementedError):
            wrapped_model.forward_multiscale_features(
                torch.rand(1, 3, 224, 224), layer_indices=[0]
            )
