#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import models as torchvision_models

from lightly_train._models.package import MultiScaleFeaturePackage
from lightly_train._models.torchvision.convnext import ConvNeXtModelWrapper
from lightly_train._models.torchvision.resnet import ResNetModelWrapper
from lightly_train._models.torchvision.shufflenet import ShuffleNetV2ModelWrapper
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper
from lightly_train._models.torchvision.torchvision_package import TorchvisionPackage

from ...helpers import DummyCustomModel


class TestTorchvisionPackage:
    @pytest.mark.parametrize(
        "model_name",
        [
            "torchvision/resnet18",
            "torchvision/convnext_small",
            "torchvision/shufflenet_v2_x0_5",
        ],
    )
    def test_list_model_names(self, model_name: str) -> None:
        assert model_name in TorchvisionPackage.list_model_names()

    def test__multiscale_feature_package(self) -> None:
        # All torchvision model wrappers support multi-scale features, which makes the
        # package available for tasks that require them.
        assert issubclass(TorchvisionPackage, MultiScaleFeaturePackage)

    @pytest.mark.parametrize(
        "model_name, wrapper_cls",
        [
            ("resnet18", ResNetModelWrapper),
            ("convnext_tiny", ConvNeXtModelWrapper),
            ("shufflenet_v2_x0_5", ShuffleNetV2ModelWrapper),
        ],
    )
    def test_get_model_wrapper(
        self, model_name: str, wrapper_cls: type[TorchvisionModelWrapper]
    ) -> None:
        model = torchvision_models.get_model(model_name, weights=None)
        assert isinstance(TorchvisionPackage.get_model_wrapper(model), wrapper_cls)

    def test_is_supported_model__true(self) -> None:
        model = torchvision_models.resnet18()
        assert TorchvisionPackage.is_supported_model(model)

        wrapped_model = TorchvisionPackage.get_model_wrapper(model=model)
        assert TorchvisionPackage.is_supported_model(wrapped_model)

    def test_is_supported_model__false(self) -> None:
        model = DummyCustomModel()
        assert not TorchvisionPackage.is_supported_model(model=model)
        assert not TorchvisionPackage.is_supported_model(model=model.get_model())

    def test_export_model(self, tmp_path: Path) -> None:
        model = torchvision_models.resnet18()
        out_path = tmp_path / "model.pt"
        TorchvisionPackage.export_model(model=model, out=out_path)
        assert out_path.exists()

        exported_model = torchvision_models.resnet18()
        exported_model.load_state_dict(
            torch.load(out_path, weights_only=True), strict=True
        )
