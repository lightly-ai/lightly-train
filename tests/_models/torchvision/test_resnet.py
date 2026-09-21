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

    @pytest.mark.parametrize(
        "model_name, dims",
        [
            # Covers both the BasicBlock of resnet18 and the Bottleneck of resnet50.
            ("resnet18", [64, 128, 256, 512]),
            ("resnet50", [256, 512, 1024, 2048]),
        ],
    )
    def test_multiscale_feature_dims(self, model_name: str, dims: list[int]) -> None:
        model = models.get_model(model_name, weights=None)
        feature_extractor = ResNetModelWrapper(model=model)
        assert feature_extractor.multiscale_feature_dims() == dims

    @pytest.mark.parametrize(
        "model_name, model_args, strides",
        [
            ("resnet18", {}, [4, 8, 16, 32]),
            ("resnet50", {}, [4, 8, 16, 32]),
            # Stages built with dilation instead of stride keep the resolution of the
            # previous stage.
            (
                "resnet50",
                {"replace_stride_with_dilation": [False, True, True]},
                [4, 8, 8, 8],
            ),
        ],
    )
    def test_multiscale_feature_strides(
        self, model_name: str, model_args: dict[str, object], strides: list[int]
    ) -> None:
        model = models.get_model(model_name, weights=None, **model_args)
        feature_extractor = ResNetModelWrapper(model=model)
        assert feature_extractor.multiscale_feature_strides() == strides

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

    def test_forward_multiscale_features__order_and_duplicates(self) -> None:
        # Features are returned in the same order as the requested indices and indices
        # can be requested more than once.
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_multiscale_features(
            x, layer_indices=[3, 1, 3]
        )
        shapes = [feature["features"].shape for feature in features]
        assert shapes == [(1, 512, 7, 7), (1, 128, 28, 28), (1, 512, 7, 7)]

    def test_forward_multiscale_features__no_indices(self) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        assert feature_extractor.forward_multiscale_features(x, layer_indices=[]) == []

    @pytest.mark.parametrize("layer_index", [-1, 4])
    def test_forward_multiscale_features__invalid_index(self, layer_index: int) -> None:
        model = models.resnet18()
        feature_extractor = ResNetModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with pytest.raises(
            ValueError,
            match=f"Layer index {layer_index} is out of range, it must be in ",
        ):
            feature_extractor.forward_multiscale_features(
                x, layer_indices=[layer_index]
            )

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
