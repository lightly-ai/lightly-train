#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from pathlib import Path

import pytest
import torch
from pytest_mock import MockerFixture

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
from lightly_train._models.dinov2_vit.dinov2_vit_src import dinov2_helper
from lightly_train._models.dinov2_vit.dinov2_vit_src.configs import (
    MODELS,
    load_and_merge_config,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
    _vit_test,
    vit_so400m,
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
            ("dinov2/vitb14-tipsv2", True),
            ("dinov2/vitl14-tipsv2", True),
            ("dinov2/vitso400m14-tipsv2", True),
            ("dinov2/vitg14-tipsv2", True),
            ("dinov2/vits14-notpretrained", True),
            ("dinov2/vitb14-notpretrained", True),
            ("dinov2/vitl14-notpretrained", True),
            ("dinov2/vitg14-notpretrained", True),
            # These models are not listed: noreg variants are secondary options and
            # noreg-notpretrained / other variants are for advanced use only.
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
            ("dinov2/vitso14-tipsv2", False),
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

    def test_tipsv2_model_names(self) -> None:
        assert DINOv2ViTPackage.parse_model_name("vitso400m14-tipsv2") == (
            "vitso400m14-tipsv2"
        )
        assert "vitso14-tipsv2" not in MODELS
        with pytest.raises(ValueError, match="Unknown model"):
            DINOv2ViTPackage.parse_model_name("vitso14-tipsv2")

    @pytest.mark.parametrize(
        ("model_name", "arch"),
        [
            ("vitb14-tipsv2", "vit_base"),
            ("vitl14-tipsv2", "vit_large"),
            ("vitso400m14-tipsv2", "vit_so400m"),
            ("vitg14-tipsv2", "vit_giant2"),
        ],
    )
    def test_tipsv2_model_configs(
        self,
        model_name: str,
        arch: str,
    ) -> None:
        config = load_and_merge_config(MODELS[model_name]["config"])
        assert config.student.arch == arch
        assert config.student.patch_size == 14
        assert config.student.num_register_tokens == 1
        assert config.student.layerscale == 1.0
        assert config.crops.global_crops_size == 448

    def test_vit_so400m_architecture(self, mocker: MockerFixture) -> None:
        constructor = mocker.patch(
            "lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer.DinoVisionTransformer"
        )

        vit_so400m()

        assert constructor.call_args.kwargs["embed_dim"] == 1152
        assert constructor.call_args.kwargs["depth"] == 27
        assert constructor.call_args.kwargs["num_heads"] == 16
        assert constructor.call_args.kwargs["mlp_ratio"] == 4304 / 1152

    def test_load_weights__pytorch_checkpoint(self, tmp_path: Path) -> None:
        expected = _vit_test()
        torch.save(expected.state_dict(), tmp_path / "tipsv2.pt")
        actual = _vit_test()

        dinov2_helper.load_weights(
            model=actual,
            checkpoint_dir=tmp_path,
            url="https://example.com/tipsv2.pt",
        )

        for expected_param, actual_param in zip(
            expected.parameters(), actual.parameters()
        ):
            assert torch.equal(expected_param, actual_param)

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

    def test_export_model__log_example_format(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that the log example shows the correct model name format (without prefix)."""
        model = DINOv2ViTPackage.get_model("_vittest14")
        out_path = tmp_path / "model.pt"
        with caplog.at_level(logging.INFO):
            DINOv2ViTPackage.export_model(model=model, out=out_path, log_example=True)

        log_output = caplog.text
        # The log should show a format without the 'dinov2/' prefix
        assert "get_model('<vitXX>')" in log_output
        assert "get_model('dinov2/" not in log_output
