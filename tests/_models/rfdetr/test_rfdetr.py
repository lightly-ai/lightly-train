#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from importlib import util as importlib_util

import pytest
import torch

from lightly_train._models.rfdetr.rfdetr import RFDETRModelWrapper

if importlib_util.find_spec("rfdetr") is None:
    pytest.skip("rfdetr is not installed", allow_module_level=True)

from rfdetr.detr import RFDETRBase


class TestRFDETRModelWrapper:
    def test_init(self) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        wrapped_model = RFDETRModelWrapper(model=model)

        for name, param in wrapped_model.named_parameters():
            assert param.requires_grad, name

        for name, module in wrapped_model.named_modules():
            assert module.training, name

    def test_feature_dim(self) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]

        wrapped_model = RFDETRModelWrapper(model=model)

        assert wrapped_model.feature_dim() == 384

    # TODO (Lionel, 05/25): remove this test when MPS is supported
    @pytest.mark.skipif(
        torch.backends.mps.is_available(), reason="MPS does not support this test"
    )
    def test_forward_features(
        self,
    ) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        device = model.model.device

        wrapped_model = RFDETRModelWrapper(model=model)

        image_size = 224
        expected_dim = wrapped_model.feature_dim()
        x = torch.rand(1, 3, image_size, image_size).to(device=device)
        features = wrapped_model.forward_features(x)["features"]

        assert features.shape == (
            1,
            expected_dim,
            int(image_size // 14),  # we use vit-14 as the backbone
            int(image_size // 14),
        )

    def test_forward_pool(self) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        device = model.model.device

        wrapped_model = RFDETRModelWrapper(model=model)

        expected_dim = wrapped_model.feature_dim()
        x = torch.rand(1, expected_dim, 7, 7).to(device=device)
        pool = wrapped_model.forward_pool({"features": x})["pooled_features"]

        assert pool.shape == (1, expected_dim, 1, 1)

    def test_get_model(self) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        wrapped_model = RFDETRModelWrapper(model=model)
        model_ = wrapped_model.get_model()
        assert model_ is model

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        wrapped_model = RFDETRModelWrapper(model=model)
        wrapped_model.to("meta")
        wrapped_model.forward_features(torch.rand(1, 3, 224, 224, device="meta"))
