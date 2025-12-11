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

from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_package import DINOv3Package
from lightly_train._models.dinov3.dinov3_src.hub import backbones
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper

from ...helpers import DummyCustomModel


class TestDINOv3Package:
    @pytest.mark.parametrize(
        "model_name, listed",
        [
            ("dinov3/_vittest16", True),
            ("dinov3/_convnexttest", True),
            ("dinov3/vitt16", True),
            ("dinov3/vitt16plus", True),
            ("dinov3/vits16", True),
            ("dinov3/vits16plus", True),
            ("dinov3/vitb16", True),
            ("dinov3/vitl16", True),
            ("dinov3/vitl16-sat493m", True),
            ("dinov3/vit7b16", True),
            ("dinov3/vit7b16-sat493m", True),
            ("dinov3/convnext-tiny", True),
            ("dinov3/convnext-small", True),
            ("dinov3/convnext-base", True),
            ("dinov3/convnext-large", True),
        ],
    )
    def test_list_model_names(self, model_name: str, listed: bool) -> None:
        model_names = DINOv3Package.list_model_names()
        assert (model_name in model_names) is listed

    def test_is_supported_model__vit_true(self) -> None:
        model = backbones._dinov3_vit_test()
        assert DINOv3Package.is_supported_model(model)

    def test_is_supported_model__convnext_true(self) -> None:
        model = backbones._dinov3_convnext_test()
        assert DINOv3Package.is_supported_model(model)

    def test_is_supported_model__wrapped_vit_true(self) -> None:
        model = DINOv3ViTModelWrapper(backbones._dinov3_vit_test())
        assert DINOv3Package.is_supported_model(model)

    def test_is_supported_model__wrapped_convnext_true(self) -> None:
        model = DINOv3VConvNeXtModelWrapper(backbones._dinov3_convnext_test())
        assert DINOv3Package.is_supported_model(model)

    def test_is_supported_model__model_false(self) -> None:
        model = DummyCustomModel().get_model()
        assert not DINOv3Package.is_supported_model(model)

    def test_is_supported_model__wrapped_model_false(self) -> None:
        model = DummyCustomModel()
        assert not DINOv3Package.is_supported_model(model)

    def test_get_model__vit(self) -> None:
        model = DINOv3Package.get_model("_vittest16")
        assert isinstance(model, DinoVisionTransformer)

    def test_get_model__convnext(self) -> None:
        model = DINOv3Package.get_model("_convnexttest")
        assert isinstance(model, ConvNeXt)

    def test_get_model_wrapper__vit(self) -> None:
        model = backbones._dinov3_vit_test()
        model_wrapper = DINOv3Package.get_model_wrapper(model=model)
        assert isinstance(model_wrapper, DINOv3ViTModelWrapper)

    def test_get_model_wrapper__convnext(self) -> None:
        model = backbones._dinov3_convnext_test()
        model_wrapper = DINOv3Package.get_model_wrapper(model=model)
        assert isinstance(model_wrapper, DINOv3VConvNeXtModelWrapper)

    @pytest.mark.parametrize(
        "model_name",
        ["_vittest16", "_convnexttest"],
    )
    def test_export_model__model(self, model_name: str, tmp_path: Path) -> None:
        model = DINOv3Package.get_model(model_name)
        out_path = tmp_path / "model.pt"
        DINOv3Package.export_model(model=model, out=out_path, log_example=False)

        model_exported = DINOv3Package.get_model(model_name)
        model_exported.load_state_dict(torch.load(out_path, weights_only=True))

        assert len(list(model.parameters())) == len(list(model_exported.parameters()))
        for (name, param), (name_exp, param_exp) in zip(
            model.named_parameters(), model_exported.named_parameters()
        ):
            assert name == name_exp
            assert param.dtype == param_exp.dtype
            assert param.requires_grad == param_exp.requires_grad
            assert torch.allclose(param, param_exp, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize(
        "model_name",
        ["_vittest16", "_convnexttest"],
    )
    def test_export_model__wrapped_model(self, model_name: str, tmp_path: Path) -> None:
        model = DINOv3Package.get_model(model_name=model_name)
        wrapped_model: DINOv3ViTModelWrapper | DINOv3VConvNeXtModelWrapper
        if isinstance(model, ConvNeXt):
            wrapped_model = DINOv3VConvNeXtModelWrapper(model=model)
        else:
            wrapped_model = DINOv3ViTModelWrapper(model=model)
        out_path = tmp_path / "model.pt"
        DINOv3Package.export_model(model=wrapped_model, out=out_path, log_example=False)

        model_exported = DINOv3Package.get_model(model_name=model_name)
        model_exported.load_state_dict(torch.load(out_path, weights_only=True))

        assert len(list(model.parameters())) == len(list(model_exported.parameters()))
        for (name, param), (name_exp, param_exp) in zip(
            model.named_parameters(), model_exported.named_parameters()
        ):
            assert name == name_exp
            assert param.dtype == param_exp.dtype
            assert param.requires_grad == param_exp.requires_grad
            assert torch.allclose(param, param_exp, rtol=1e-3, atol=1e-4)

    def test_export_model__unsupported_model(self, tmp_path: Path) -> None:
        model = DummyCustomModel().get_model()
        out_path = tmp_path / "model.pt"
        with pytest.raises(ValueError):
            DINOv3Package.export_model(model=model, out=out_path)
