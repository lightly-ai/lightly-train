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

try:
    import timm
except ImportError:
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("timm is not installed", allow_module_level=True)

from lightly_train._models.timm.timm_package import TIMMPackage

from ...helpers import DummyCustomModel


class TestTIMMPackage:
    def test_is_supported_model__true(self) -> None:
        model = timm.create_model("resnet18")
        assert TIMMPackage.is_supported_model(model)
        assert TIMMPackage.is_supported_model(
            TIMMPackage.get_model_wrapper(model=model)
        )

    def test_is_supported_model__false(self) -> None:
        model = DummyCustomModel()
        assert not TIMMPackage.is_supported_model(model=model)
        assert not TIMMPackage.is_supported_model(model=model.get_model())

    def test_export_model__model_detailed(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = timm.create_model("resnet18", pretrained=False)

        TIMMPackage.export_model(model=model, out=out)
        model_exported = timm.create_model(
            "resnet18", pretrained=False, checkpoint_path=str(out)
        )

        # Check that parameters are the same.
        assert len(list(model.parameters())) == len(list(model_exported.parameters()))
        for (name, param), (name_exp, param_exp) in zip(
            model.named_parameters(), model_exported.named_parameters()
        ):
            assert name == name_exp
            assert param.dtype == param_exp.dtype
            assert param.requires_grad == param_exp.requires_grad
            assert torch.allclose(param, param_exp, rtol=1e-3, atol=1e-4)

        # Check module states.
        assert len(list(model.modules())) == len(list(model_exported.modules()))
        for (name, module), (name_exp, module_exp) in zip(
            model.named_modules(), model_exported.named_modules()
        ):
            assert name == name_exp
            assert module.training
            assert module_exp.training

    def test_export_model__wrapped_model_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = timm.create_model("resnet18", pretrained=False)
        wrapped_model = TIMMPackage.get_model_wrapper(model=model)

        TIMMPackage.export_model(model=wrapped_model, out=out)
        timm.create_model("resnet18", pretrained=False, checkpoint_path=str(out))

    def test_export_model__unsupported_model(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = DummyCustomModel()

        with pytest.raises(ValueError, match="TIMMPackage only supports"):
            TIMMPackage.export_model(model=model, out=out)

        with pytest.raises(ValueError, match="TIMMPackage only supports"):
            TIMMPackage.export_model(model=model.get_model(), out=out)
