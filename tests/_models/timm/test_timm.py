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

from lightly_train._models.model_wrapper import (
    MultiScaleFeatureCNN,
    MultiScaleFeatureViT,
)
from lightly_train._models.timm import timm as timm_feature_extractor
from lightly_train._models.timm.timm import (
    TIMMModelWrapper,
    TIMMMultiScaleCNNModelWrapper,
    TIMMMultiScaleViTModelWrapper,
    get_multiscale_model_wrapper,
)


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

        predictions = model.forward_head(extractor.forward_features(x)["features"])  # type: ignore[operator]
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


class TestTIMMMultiScaleViTModelWrapper:
    def test_patch_size(self) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        assert extractor.patch_size() == 16

    def test_multiscale_feature_dims(self) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        assert extractor.multiscale_feature_dims() == [192] * 12

    @torch.no_grad()
    def test_forward_multiscale_features(self) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = extractor.forward_multiscale_features(x, layer_indices=[0, 5, 11])
        assert [f["features"].shape for f in features] == [(1, 192, 14, 14)] * 3
        # The cls_token is optional and not returned.
        assert all("cls_token" not in f for f in features)

    @torch.no_grad()
    def test_forward_multiscale_features__dynamic_img_size(self) -> None:
        # TIMMPackage.get_model creates vit models with dynamic_img_size=True, which
        # allows inputs that differ from the image size the model was defined with.
        model = timm.create_model("vit_tiny_patch16_224", dynamic_img_size=True)
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        x = torch.rand(1, 3, 96, 96)
        features = extractor.forward_multiscale_features(x, layer_indices=[0, 11])
        assert [f["features"].shape for f in features] == [(1, 192, 6, 6)] * 2

    @torch.no_grad()
    def test_forward_multiscale_features__order(self) -> None:
        # Timm returns the intermediates in the order in which they are produced by the
        # model, independent of the order of the requested indices.
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = extractor.forward_multiscale_features(x, layer_indices=[0, 5, 11])
        features_shuffled = extractor.forward_multiscale_features(
            x, layer_indices=[11, 0, 5]
        )
        for feature, index in zip(features_shuffled, [2, 0, 1]):
            testing.assert_close(feature["features"], features[index]["features"])

    @torch.no_grad()
    def test_forward_multiscale_features__equality_to_forward_features(self) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = extractor.forward_multiscale_features(x, layer_indices=[11])
        testing.assert_close(
            features[0]["features"], extractor.forward_features(x)["features"]
        )

    @pytest.mark.parametrize("layer_index", [-1, 12])
    def test_forward_multiscale_features__invalid_index(self, layer_index: int) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        with pytest.raises(ValueError, match="is out of range"):
            extractor.forward_multiscale_features(x, layer_indices=[0, layer_index])

    @torch.no_grad()
    def test_forward_multiscale_features__no_indices(self) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        extractor = TIMMMultiScaleViTModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        assert extractor.forward_multiscale_features(x, layer_indices=[]) == []

    def test__non_constant_feature_geometry(self) -> None:
        # The wrapper must not be usable for models with per-stage feature dimensions,
        # it would report the stride of the first stage as the patch size.
        model = timm.create_model("resnet18")
        with pytest.raises(
            ValueError, match="does not have the same feature dimension"
        ):
            TIMMMultiScaleViTModelWrapper(model=model)

    def test_protocol(self) -> None:
        model = timm.create_model("vit_tiny_patch16_224")
        # The type annotation is checked by mypy.
        extractor: MultiScaleFeatureViT = TIMMMultiScaleViTModelWrapper(model=model)
        # The wrapper must not also satisfy MultiScaleFeatureCNN, otherwise consumers
        # cannot tell the two protocols apart.
        assert not hasattr(extractor, "multiscale_feature_strides")


class TestTIMMMultiScaleCNNModelWrapper:
    @pytest.mark.parametrize(
        "model_name, expected_dims, expected_strides",
        [
            # ResNet has an additional entry for the stem.
            ("resnet18", [64, 64, 128, 256, 512], [2, 4, 8, 16, 32]),
            ("convnext_tiny", [96, 192, 384, 768], [4, 8, 16, 32]),
            ("swin_tiny_patch4_window7_224", [96, 192, 384, 768], [4, 8, 16, 32]),
        ],
    )
    def test_multiscale_feature_dims_and_strides(
        self, model_name: str, expected_dims: list[int], expected_strides: list[int]
    ) -> None:
        model = timm.create_model(model_name)
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        assert extractor.multiscale_feature_dims() == expected_dims
        assert extractor.multiscale_feature_strides() == expected_strides

    @torch.no_grad()
    @pytest.mark.parametrize(
        "model_name, image_size",
        [
            ("resnet18", 64),
            ("convnext_tiny", 64),
            ("swin_tiny_patch4_window7_224", 224),
        ],
    )
    def test_forward_multiscale_features(
        self, model_name: str, image_size: int
    ) -> None:
        model = timm.create_model(model_name)
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        dims = extractor.multiscale_feature_dims()
        strides = extractor.multiscale_feature_strides()
        assert len(dims) == len(strides)

        x = torch.rand(1, 3, image_size, image_size)
        layer_indices = list(range(len(dims)))
        features = extractor.forward_multiscale_features(x, layer_indices=layer_indices)
        assert [f["features"].shape for f in features] == [
            (1, dim, image_size // stride, image_size // stride)
            for dim, stride in zip(dims, strides)
        ]
        # The cls_token is optional and not returned.
        assert all("cls_token" not in f for f in features)

    @torch.no_grad()
    def test_forward_multiscale_features__order(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        features = extractor.forward_multiscale_features(x, layer_indices=[0, 2, 3])
        features_shuffled = extractor.forward_multiscale_features(
            x, layer_indices=[3, 0, 2]
        )
        for feature, index in zip(features_shuffled, [2, 0, 1]):
            testing.assert_close(feature["features"], features[index]["features"])

    @torch.no_grad()
    @pytest.mark.parametrize("model_name", ["resnet18", "convnext_tiny"])
    def test_forward_multiscale_features__equality_to_forward_features(
        self, model_name: str
    ) -> None:
        # Only holds for models whose last stage is not followed by additional layers
        # and that have no final normalization applied by forward_features.
        model = timm.create_model(model_name)
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        layer_index = len(extractor.multiscale_feature_dims()) - 1
        features = extractor.forward_multiscale_features(x, layer_indices=[layer_index])
        testing.assert_close(
            features[0]["features"], extractor.forward_features(x)["features"]
        )

    @torch.no_grad()
    def test_forward_multiscale_features__duplicate_indices(self) -> None:
        # Timm ignores duplicate indices, the wrapper must still return one entry per
        # requested index.
        model = timm.create_model("resnet18")
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        features = extractor.forward_multiscale_features(x, layer_indices=[2, 2, 0])
        assert [f["features"].shape for f in features] == [
            (1, 128, 8, 8),
            (1, 128, 8, 8),
            (1, 64, 32, 32),
        ]

    @pytest.mark.parametrize("layer_index", [-1, 5])
    def test_forward_multiscale_features__invalid_index(self, layer_index: int) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="is out of range"):
            extractor.forward_multiscale_features(x, layer_indices=[0, layer_index])

    @torch.no_grad()
    def test_forward_multiscale_features__no_indices(self) -> None:
        model = timm.create_model("resnet18")
        extractor = TIMMMultiScaleCNNModelWrapper(model=model)
        x = torch.rand(1, 3, 64, 64)
        assert extractor.forward_multiscale_features(x, layer_indices=[]) == []

    def test__non_increasing_feature_strides(self) -> None:
        # The wrapper must not be usable for models with a single feature resolution,
        # it would report the same stride for all stages.
        model = timm.create_model("vit_tiny_patch16_224")
        with pytest.raises(
            ValueError, match="does not have increasing feature strides"
        ):
            TIMMMultiScaleCNNModelWrapper(model=model)

    def test_protocol(self) -> None:
        model = timm.create_model("resnet18")
        # The type annotation is checked by mypy.
        extractor: MultiScaleFeatureCNN = TIMMMultiScaleCNNModelWrapper(model=model)
        # The wrapper must not also satisfy MultiScaleFeatureViT, otherwise consumers
        # cannot tell the two protocols apart.
        assert not hasattr(extractor, "patch_size")


@pytest.mark.parametrize(
    "model_name, expected_cls",
    [
        ("vit_tiny_patch16_224", TIMMMultiScaleViTModelWrapper),
        ("resnet18", TIMMMultiScaleCNNModelWrapper),
        ("convnext_tiny", TIMMMultiScaleCNNModelWrapper),
        # Swin is a transformer but has one feature resolution per stage.
        ("swin_tiny_patch4_window7_224", TIMMMultiScaleCNNModelWrapper),
        # MobileViT is a hybrid architecture.
        ("mobilevit_xxs", TIMMMultiScaleCNNModelWrapper),
    ],
)
def test_get_multiscale_model_wrapper(model_name: str, expected_cls: type) -> None:
    model = timm.create_model(model_name)
    assert type(get_multiscale_model_wrapper(model=model)) is expected_cls


def test_get_multiscale_model_wrapper__no_forward_intermediates() -> None:
    model = timm.create_model("convit_tiny")
    with pytest.raises(ValueError, match="has no 'forward_intermediates' method"):
        get_multiscale_model_wrapper(model=model)


def test_get_multiscale_model_wrapper__no_feature_info() -> None:
    model = timm.create_model("resnet18")
    del model.feature_info
    with pytest.raises(ValueError, match="has no 'feature_info'"):
        get_multiscale_model_wrapper(model=model)


def test_get_multiscale_model_wrapper__mixed_geometry() -> None:
    # VOLO has feature strides [8, 16, 16, 16], which are neither constant nor
    # increasing from stage to stage.
    model = timm.create_model("volo_d1_224")
    with pytest.raises(ValueError, match="which are neither constant"):
        get_multiscale_model_wrapper(model=model)


def test_get_multiscale_model_wrapper__non_square_patch_size() -> None:
    model = timm.create_model(
        "vit_tiny_patch16_224", patch_size=(16, 8), dynamic_img_size=True
    )
    with pytest.raises(ValueError, match="has a non-square patch size"):
        get_multiscale_model_wrapper(model=model)


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
    expected = model.forward_features(x)  # type: ignore[operator]
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
    features = model.forward_features(x)  # type: ignore[operator]
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


def test_architecture_info__all_timm_model_names_match_prefix() -> None:
    """Verify that every timm model name matches at least one known prefix.

    This test does not instantiate any models, so it is fast. It ensures no
    model family has been forgotten in _TIMM_ARCH_NAME_PREFIXES.
    """
    prefixes = [prefix for prefix, _ in timm_feature_extractor._TIMM_ARCH_NAME_PREFIXES]
    unmatched = [
        name
        for name in timm.list_models()
        if not any(name.startswith(prefix) for prefix in prefixes)
    ]
    assert not unmatched, (
        f"{len(unmatched)} timm model(s) have no matching prefix in "
        f"_TIMM_ARCH_NAME_PREFIXES: {unmatched}"
    )
