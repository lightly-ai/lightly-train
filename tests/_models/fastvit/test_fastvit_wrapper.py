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

from lightly_train._models.fastvit.fastvit import FastViTModelWrapper

from ...helpers import dummy_fastvit_model


@pytest.fixture
def wrapper() -> FastViTModelWrapper:
    return dummy_fastvit_model()


class TestFastViTModelWrapper:
    def test_multiscale_feature_dims(self, wrapper: FastViTModelWrapper) -> None:
        dims = wrapper.multiscale_feature_dims()
        assert len(dims) == 4
        # fastvit_t8 has embed_dims = [48, 96, 192, 384]
        assert dims == [48, 96, 192, 384]

    def test_multiscale_feature_dims__cached(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        dims1 = wrapper.multiscale_feature_dims()
        dims2 = wrapper.multiscale_feature_dims()
        assert dims1 is dims2

    def test_multiscale_feature_strides(self, wrapper: FastViTModelWrapper) -> None:
        strides = wrapper.multiscale_feature_strides()
        assert len(strides) == 4
        assert strides == [4, 8, 16, 32]

    def test_multiscale_feature_strides__cached(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        strides1 = wrapper.multiscale_feature_strides()
        strides2 = wrapper.multiscale_feature_strides()
        assert strides1 is strides2

    def test_forward_multiscale_features__all_stages(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(1, 3, 64, 64)
        feats = wrapper.forward_multiscale_features(x, [0, 1, 2, 3])
        assert len(feats) == 4

        dims = wrapper.multiscale_feature_dims()
        strides = wrapper.multiscale_feature_strides()
        for i, feat in enumerate(feats):
            assert "features" in feat
            assert feat["features"].shape[1] == dims[i]
            expected_h = 64 // strides[i]
            expected_w = 64 // strides[i]
            assert feat["features"].shape[-2:] == (expected_h, expected_w)

    def test_forward_multiscale_features__subset(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(2, 3, 64, 64)
        feats = wrapper.forward_multiscale_features(x, [1, 3])
        assert len(feats) == 2

        dims = wrapper.multiscale_feature_dims()
        strides = wrapper.multiscale_feature_strides()
        assert feats[0]["features"].shape == (
            2,
            dims[1],
            64 // strides[1],
            64 // strides[1],
        )
        assert feats[1]["features"].shape == (
            2,
            dims[3],
            64 // strides[3],
            64 // strides[3],
        )

    def test_forward_multiscale_features__invalid_index_raises(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(1, 3, 64, 64)
        with pytest.raises(AssertionError):
            wrapper.forward_multiscale_features(x, [4])

    def test_forward_multiscale_features__negative_index_raises(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(1, 3, 64, 64)
        with pytest.raises(AssertionError):
            wrapper.forward_multiscale_features(x, [-1])
