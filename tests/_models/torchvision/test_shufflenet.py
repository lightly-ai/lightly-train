#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

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

    @pytest.mark.parametrize(
        "model_name, dims",
        [
            # The last stage is conv5, which expands the feature dimension to a width
            # that does not depend on the model size in the same way as the stages.
            ("shufflenet_v2_x0_5", [48, 96, 192, 1024]),
            ("shufflenet_v2_x2_0", [244, 488, 976, 2048]),
        ],
    )
    def test_multiscale_feature_dims(self, model_name: str, dims: list[int]) -> None:
        model = models.get_model(model_name, weights=None)
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        assert wrapped_model.multiscale_feature_dims() == dims

    def test_multiscale_feature_strides(self) -> None:
        # conv5 does not downsample, so it has the same stride as stage4.
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        assert wrapped_model.multiscale_feature_strides() == [8, 16, 32, 32]

    def test_forward_multiscale_features(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = wrapped_model.forward_multiscale_features(
            x, layer_indices=[0, 1, 2, 3]
        )
        shapes = [feature["features"].shape for feature in features]
        assert shapes == [
            (1, 48, 28, 28),
            (1, 96, 14, 14),
            (1, 192, 7, 7),
            (1, 1024, 7, 7),
        ]

    def test_forward_multiscale_features__order_and_duplicates(self) -> None:
        # Features are returned in the same order as the requested indices and indices
        # can be requested more than once.
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = wrapped_model.forward_multiscale_features(x, layer_indices=[3, 1, 3])
        shapes = [feature["features"].shape for feature in features]
        assert shapes == [(1, 1024, 7, 7), (1, 96, 14, 14), (1, 1024, 7, 7)]

    def test_forward_multiscale_features__no_indices(self) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        assert wrapped_model.forward_multiscale_features(x, layer_indices=[]) == []

    @pytest.mark.parametrize("layer_index", [-1, 4])
    def test_forward_multiscale_features__invalid_index(self, layer_index: int) -> None:
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with pytest.raises(
            ValueError,
            match=f"Layer index {layer_index} is out of range, it must be in ",
        ):
            wrapped_model.forward_multiscale_features(x, layer_indices=[layer_index])

    def test_forward_multiscale_features__matches_forward_features(self) -> None:
        # The last stage is conv5 and therefore matches forward_features. Eval mode
        # disables stochastic layers so that the two forward passes are deterministic.
        model = models.shufflenet_v2_x0_5()
        wrapped_model = ShuffleNetV2ModelWrapper(model=model).eval()
        x = torch.rand(1, 3, 224, 224)
        last = wrapped_model.forward_multiscale_features(x, layer_indices=[3])[0]
        assert torch.equal(
            last["features"], wrapped_model.forward_features(x)["features"]
        )
