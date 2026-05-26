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

try:
    import fastervit
except ImportError:
    pytest.skip("fastervit is not installed", allow_module_level=True)

from lightly_train._models.fastervit.fastervit import FasterViTModelWrapper


class TestFasterViTModelWrapper:
    def test_forward_features(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            y = wrapper.forward_features(x)["features"]
        assert y.shape == (1, 512, 7, 7)

    def test_forward_pool(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        x = torch.rand(1, 512, 7, 7)
        y = wrapper.forward_pool({"features": x})["pooled_features"]
        assert y.shape == (1, 512, 1, 1)

    def test_get_model(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        assert wrapper.get_model() is model

    def test_forward__equality_to_model(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            predictions = model.forward_head(wrapper.forward_features(x)["features"])  # type: ignore[operator]
            predictions_direct = model(x)
        torch.testing.assert_close(predictions, predictions_direct)

    def test_forward__faster_vit_0_224__shape(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            y = wrapper.forward_pool(wrapper.forward_features(x))["pooled_features"]
        assert y.shape == (1, 512, 1, 1)

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        wrapper.to("meta")
        wrapper.forward_features(torch.rand(1, 3, 224, 224, device="meta"))

    def test_feature_dim(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        assert wrapper.feature_dim() == 512

    def test_architecture_info__batchnorm(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapper = FasterViTModelWrapper(model=model)
        info = wrapper.architecture_info()
        assert info == {"model_type": "hybrid", "norm_type": "batchnorm"}
