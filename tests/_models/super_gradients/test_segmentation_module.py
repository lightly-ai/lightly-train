#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

try:
    import super_gradients  # noqa: F401
except ImportError:
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("super_gradients is not installed", allow_module_level=True)


from lightly_train._models.super_gradients.segmentation_module import (
    SegmentationModuleModelWrapper,
)
from lightly_train._models.super_gradients.super_gradients_package import (
    SUPER_GRADIENTS_PACKAGE,
)


class TestSegmentationModule:
    def test_feature_dim(self) -> None:
        model = SUPER_GRADIENTS_PACKAGE.get_model("pp_lite_t_seg50")
        fe = SegmentationModuleModelWrapper(model)
        assert fe.feature_dim() == 1024

    def test_forward_features(self) -> None:
        model = SUPER_GRADIENTS_PACKAGE.get_model("pp_lite_t_seg50")
        fe = SegmentationModuleModelWrapper(model)
        x = torch.rand(1, 3, 224, 224)
        out = fe.forward_features(x)["features"]
        assert out.shape == (1, 1024, 7, 7)

    def test_forward_pool(self) -> None:
        model = SUPER_GRADIENTS_PACKAGE.get_model("pp_lite_t_seg50")
        fe = SegmentationModuleModelWrapper(model)
        x = torch.rand(1, 3, 224, 224)
        out = fe.forward_pool({"features": x})["pooled_features"]
        assert out.shape == (1, 3, 1, 1)

    def test_get_model(self) -> None:
        model = SUPER_GRADIENTS_PACKAGE.get_model("pp_lite_t_seg50")
        fe = SegmentationModuleModelWrapper(model)
        assert fe.get_model() is model

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = SUPER_GRADIENTS_PACKAGE.get_model("pp_lite_t_seg50")
        fe = SegmentationModuleModelWrapper(model)
        fe.to("meta")
        fe.forward_features(torch.rand(1, 3, 224, 224, device="meta"))
