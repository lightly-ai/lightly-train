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
    def test_is_model(self) -> None:
        model = timm.create_model("resnet18")
        assert TIMMPackage.is_supported_model(model)

    def test_is_model__false(self) -> None:
        model = DummyCustomModel()
        assert not TIMMPackage.is_supported_model(model)

    def test_export_model(self, tmp_path: Path) -> None:
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
