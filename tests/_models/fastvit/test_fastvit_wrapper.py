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
from lightly_train._models.fastvit.fastvit_package import FASTVIT_PACKAGE
from lightly_train._models.package import MultiScaleFeaturePackage
from lightly_train._task_models.linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)

from ...helpers import dummy_fastvit_model


@pytest.fixture
def wrapper() -> FastViTModelWrapper:
    return dummy_fastvit_model()


@pytest.fixture
def fork_feature_wrapper() -> FastViTModelWrapper:
    return dummy_fastvit_model(fork_feat=True)


class TestFastViTModelWrapper:
    def test_package_is_multiscale_feature_package(self) -> None:
        assert isinstance(FASTVIT_PACKAGE, MultiScaleFeaturePackage)
        assert LinearSemanticSegmentation.is_supported_model("fastvit/fastvit_t8-linear")

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

    def test_forward_features_and_pool__classifier_model(
        self, wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(1, 3, 64, 64)
        features = wrapper.forward_features(x)
        pooled = wrapper.forward_pool(features)

        assert features["features"].shape == (1, wrapper.feature_dim(), 2, 2)
        assert pooled["pooled_features"].shape == (1, wrapper.feature_dim(), 1, 1)

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

    def test_fork_feature_model__forward_features_and_pool(
        self, fork_feature_wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(1, 3, 64, 64)
        features = fork_feature_wrapper.forward_features(x)
        pooled = fork_feature_wrapper.forward_pool(features)

        assert features["features"].shape == (1, 384, 2, 2)
        assert pooled["pooled_features"].shape == (1, 384, 1, 1)

    def test_fork_feature_model__uses_normalized_stage_outputs(
        self, fork_feature_wrapper: FastViTModelWrapper
    ) -> None:
        x = torch.randn(1, 3, 64, 64)
        model = fork_feature_wrapper.get_model().eval()
        fork_feature_wrapper.eval()

        with torch.no_grad():
            expected = model(x)
            actual = fork_feature_wrapper.forward_multiscale_features(x, [0, 1, 2, 3])
            final_features = fork_feature_wrapper.forward_features(x)

        assert isinstance(expected, list)
        for expected_feature, actual_feature in zip(expected, actual):
            torch.testing.assert_close(actual_feature["features"], expected_feature)
        torch.testing.assert_close(final_features["features"], expected[-1])
