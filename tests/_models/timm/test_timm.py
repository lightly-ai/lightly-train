#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Callable

import pytest
import torch
from torch import Tensor, testing
from torch.nn import Module

try:
    import timm
except ImportError:
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("timm is not installed", allow_module_level=True)

from lightly_train._models.timm import timm as timm_feature_extractor
from lightly_train._models.timm.timm import TIMMModelWrapper


class TestTIMMModelWrapper:
    def test_forward_features(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        y = extractor.forward_features(x)["features"]
        assert y.shape == (1, 512, 2, 2)

    def test_forward_pool(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMModelWrapper(model=model)
        x = torch.rand(1, 32, 2, 2)
        y = extractor.forward_pool({"features": x})["pooled_features"]
        assert y.shape == (1, 32, 1, 1)

    def test_get_model(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMModelWrapper(model=model)
        assert extractor.get_model() is model

    def test_forward__equality_to_model(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)

        predictions = model.forward_head(extractor.forward_features(x)["features"])
        predictions_direct = model(x)

        torch.testing.assert_close(predictions, predictions_direct)

    def test_forward__resnet18__shape(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        y = extractor.forward_pool(extractor.forward_features(x))["pooled_features"]
        assert y.shape == (1, 512, 1, 1)

    def test_forward__flexivit_small__shape(self) -> None:
        model = timm.create_model("flexivit_small")
        extractor = TIMMModelWrapper(model=model)
        x = torch.rand(1, 3, 240, 240)
        y = extractor.forward_pool(extractor.forward_features(x))["pooled_features"]
        assert y.shape == (1, 384, 1, 1)

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = timm.create_model("resnet18")
        extractor = TIMMModelWrapper(model=model)
        extractor.to("meta")
        extractor.forward_features(torch.rand(1, 3, 64, 64, device="meta"))


# TODO: Do not skip if timm <1.0
@pytest.mark.skip(reason="Requires timm <1.0")
def test_get_forward_features_fn__forward_features() -> None:
    model = timm.create_model("resnet18")
    assert (
        timm_feature_extractor._get_forward_features_fn(model=model)
        == timm_feature_extractor._forward_features
    )


# TODO: Do not skip if timm <1.0 and >=0.9
@pytest.mark.skip(reason="Requires timm <1.0 and >=0.9")
def test_get_forward_features_fn__get_intermediate_layers() -> None:
    model = timm.create_model("vit_tiny_patch16_224")
    assert (
        timm_feature_extractor._get_forward_features_fn(model=model)
        == timm_feature_extractor._forward_get_intermediate_layers
    )


def test_get_forward_featres_fn__forward_intermediates() -> None:
    model = timm.create_model("vit_tiny_patch16_224")
    assert (
        timm_feature_extractor._get_forward_features_fn(model=model)
        == timm_feature_extractor._forward_intermediates
    )


# After timm >= 1.0 all models should have forward_intermediates method.
@torch.no_grad()
@pytest.mark.parametrize(
    "fn, method_name",
    [
        (timm_feature_extractor._forward_features, "forward_features"),
        (timm_feature_extractor._forward_intermediates, "forward_intermediates"),
        (
            timm_feature_extractor._forward_get_intermediate_layers,
            "get_intermediate_layers",
        ),
    ],
)
def test__forward_features(
    fn: Callable[[Module, Tensor], Tensor], method_name: str
) -> None:
    model = timm.create_model("vit_tiny_patch16_224", class_token=True, reg_tokens=2)
    # Not all models and timm versions have forward_intermediates and
    # get_intermediate_layers method defined.
    if method_name != "forward_features" and not hasattr(model, method_name):
        pytest.skip(f"Model does not have '{method_name}' method")

    x = torch.rand(1, 3, 224, 224)
    features = fn(model, x)
    assert features.shape == (1, 192, 14, 14)
    expected = model.forward_features(x)
    expected = expected[:, 3:]  # Drop class token + 2 reg tokens
    expected = timm_feature_extractor._to_nchw(expected)
    testing.assert_close(features, expected)


@torch.no_grad()
@pytest.mark.parametrize(
    "class_token, reg_tokens, global_pool",
    [(False, 0, "avg"), (True, 0, "token"), (True, 2, "token")],
)
def test__drop_prefix_tokens(
    class_token: bool, reg_tokens: int, global_pool: str
) -> None:
    model = timm.create_model(
        "vit_tiny_patch16_224",
        class_token=class_token,
        reg_tokens=reg_tokens,
        global_pool=global_pool,
    )
    x = torch.rand(1, 3, 224, 224)
    features = model.forward_features(x)
    features = timm_feature_extractor._drop_prefix_tokens(model, features)
    assert features.shape == (1, 14 * 14, 192)


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((1, 64, 8, 8), (1, 64, 8, 8)),
        ((1, 192, 14, 14), (1, 192, 14, 14)),
        ((1, 8 * 8, 64), (1, 64, 8, 8)),
        ((1, 14 * 14, 192), (1, 192, 14, 14)),
    ],
)
def test__to_nchw(shape: tuple[int, ...], expected: tuple[int, ...]) -> None:
    x = torch.rand(shape)
    y = timm_feature_extractor._to_nchw(x)
    assert y.shape == expected
