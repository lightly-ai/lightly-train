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

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
    _vit_test,
)

from ...helpers import DummyCustomModel


class TestDINOv2ViTPackage:
    @pytest.mark.parametrize(
        "model_name, listed",
        [
            ("dinov2/vits14", True),
            ("dinov2/vitb14", True),
            ("dinov2/vitl14", True),
            ("dinov2/vitg14", True),
            # These models are False because they are not listed but they can still be
            # instantiated if needed.
            ("dinov2/vits14-noreg", False),
            ("dinov2/vitb14-noreg", False),
            ("dinov2/vitl14-noreg", False),
            ("dinov2/vitg14-noreg", False),
            ("dinov2/vits14-noreg-pretrained", False),
            ("dinov2/vitb14-noreg-pretrained", False),
            ("dinov2/vitl14-noreg-pretrained", False),
            ("dinov2/vitg14-noreg-pretrained", False),
            ("dinov2/vits14-pretrained", False),
            ("dinov2/vitb14-pretrained", False),
            ("dinov2/vitl14-pretrained", False),
            ("dinov2/vitg14-pretrained", False),
            # The dinov2_vit/XYZ models are False because they are deprecated. You can
            # still instantiate them but they are not in the list of model names.
            ("dinov2_vit/_vittest14", False),
            ("dinov2_vit/vits14", False),
            ("dinov2_vit/vitb14", False),
            ("dinov2_vit/vitl14", False),
            ("dinov2_vit/vitg14", False),
            ("dinov2/_vittest14", False),
        ],
    )
    def test_list_model_names(self, model_name: str, listed: bool) -> None:
        model_names = DINOv2ViTPackage.list_model_names()
        assert (model_name in model_names) is listed

    def test_is_supported_model__model_true(self) -> None:
        model = _vit_test()
        assert DINOv2ViTPackage.is_supported_model(model)

    def test_is_supported_model__wrapped_model_true(self) -> None:
        model = _vit_test()
        wrapped_model = DINOv2ViTModelWrapper(model=model)
        assert DINOv2ViTPackage.is_supported_model(wrapped_model)

    def test_is_supported_model__model_false(self) -> None:
        model = DummyCustomModel().get_model()
        assert not DINOv2ViTPackage.is_supported_model(model)

    def test_is_supported_model__wrapped_model_false(self) -> None:
        model = DummyCustomModel()
        assert not DINOv2ViTPackage.is_supported_model(model)

    @pytest.mark.parametrize(
        "model_name",
        [
            "_vittest14",
            "vits14-notpretrained",
            "vits14-noreg-notpretrained",
            "vitb14-noreg-notpretrained",
        ],
    )
    def test_get_model(self, model_name: str) -> None:
        model = DINOv2ViTPackage.get_model(model_name=model_name)
        assert isinstance(model, DinoVisionTransformer)

    def test_get_model_wrapper(self) -> None:
        model = _vit_test()
        fe = DINOv2ViTPackage.get_model_wrapper(model=model)
        assert isinstance(fe, DINOv2ViTModelWrapper)

    @pytest.mark.parametrize(
        "model_name",
        ["_vittest14"],
    )
    def test_export_model__model(self, model_name: str, tmp_path: Path) -> None:
        model = DINOv2ViTPackage.get_model(model_name)
        out_path = tmp_path / "model.pt"
        DINOv2ViTPackage.export_model(model=model, out=out_path, log_example=False)

        model_exported = DINOv2ViTPackage.get_model(model_name)
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
        ["_vittest14"],
    )
    def test_export_model__wrapped_model(self, model_name: str, tmp_path: Path) -> None:
        model = DINOv2ViTPackage.get_model(model_name=model_name)
        wrapped_model = DINOv2ViTModelWrapper(model=model)
        out_path = tmp_path / "model.pt"
        DINOv2ViTPackage.export_model(
            model=wrapped_model, out=out_path, log_example=False
        )

        model_exported = DINOv2ViTPackage.get_model(model_name=model_name)
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
            DINOv2ViTPackage.export_model(model=model, out=out_path)
