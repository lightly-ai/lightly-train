#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from pytest_mock import MockerFixture
from torch import Tensor
from torch.nn import Module

from lightly_train._models.custom.custom_package import CustomPackage


class TestCustomPackage:
    def test_is_supported_model(self) -> None:
        class DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

        model = DummyCustomModel()
        assert CustomPackage.is_supported_model(model)

    def test_is_supported_model__no_feature_dim(self, mocker: MockerFixture) -> None:
        class DummyCustomModel(Module):
            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

        model = DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)

    def test_is_supported_model__no_forward_features(
        self, mocker: MockerFixture
    ) -> None:
        class DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

        model = DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)

    def test_is_custom_model__no_forward_pool(self, mocker: MockerFixture) -> None:
        class DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

        model = DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)
