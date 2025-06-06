#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor
from torch.nn import Module

from lightly_train._models.custom.custom_package import CustomPackage

from ...helpers import DummyCustomModel


class TestCustomPackage:
    def test_is_supported_model(self) -> None:
        class _DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def get_model(self) -> Module:
                return self

        model = _DummyCustomModel()
        assert CustomPackage.is_supported_model(model)

    def test_is_supported_model__no_feature_dim(self) -> None:
        class _DummyCustomModel(Module):
            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def get_model(self) -> Module:
                return self

        model = _DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)

    def test_is_supported_model__no_forward_features(self) -> None:
        class _DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def get_model(self) -> Module:
                return self

        model = _DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)

    def test_is_custom_model__no_forward_pool(self, mocker: MockerFixture) -> None:
        class _DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def get_model(self) -> Module:
                return self

        model = _DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)

    def test_is_custom_model__no_get_model(self, mocker: MockerFixture) -> None:
        class _DummyCustomModel(Module):
            def feature_dim(self) -> int:
                return 1

            def forward_features(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

            def forward_pool(self, x: Tensor) -> Tensor:
                return torch.zeros(1)

        model = _DummyCustomModel()
        assert not CustomPackage.is_supported_model(model)

    def test_export_model__wrapped_model(self, tmp_path: Path) -> None:
        model = DummyCustomModel()
        out = tmp_path / "model.pth"
        CustomPackage.export_model(model, out)
        assert out.exists()

        model.get_model().load_state_dict(
            torch.load(out, weights_only=True), strict=True
        )

    def test_export_model__unwrapped_model(self, tmp_path: Path) -> None:
        model = DummyCustomModel()
        out = tmp_path / "model.pth"
        CustomPackage.export_model(model=model.get_model(), out=out)
        assert out.exists()

        model.get_model().load_state_dict(
            torch.load(out, weights_only=True), strict=True
        )

    def test_export_model__unsupported_model(self, tmp_path: Path) -> None:
        class UnsupportedModel:
            pass

        model = UnsupportedModel()
        out = tmp_path / "model.pth"
        with pytest.raises(ValueError):
            CustomPackage.export_model(model, out)
