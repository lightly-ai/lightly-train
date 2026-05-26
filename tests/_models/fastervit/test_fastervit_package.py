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

try:
    import fastervit
except ImportError:
    pytest.skip("fastervit is not installed", allow_module_level=True)

from lightly_train._models.fastervit.fastervit_package import FasterViTPackage

from ...helpers import DummyCustomModel


class TestFasterViTPackage:
    def test_is_supported_model__true(self) -> None:
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        assert FasterViTPackage.is_supported_model(model)
        assert FasterViTPackage.is_supported_model(
            FasterViTPackage.get_model_wrapper(model=model)
        )

    def test_is_supported_model__false(self) -> None:
        model = DummyCustomModel()
        assert not FasterViTPackage.is_supported_model(model=model)
        assert not FasterViTPackage.is_supported_model(model=model.get_model())

    def test_export_model__model_detailed(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)

        FasterViTPackage.export_model(model=model, out=out)

        # Reload and compare parameters.
        model_reloaded = fastervit.create_model("faster_vit_0_224", pretrained=False)
        model_reloaded.load_state_dict(torch.load(out, weights_only=True))

        assert len(list(model.parameters())) == len(list(model_reloaded.parameters()))
        for (name, param), (name_exp, param_exp) in zip(
            model.named_parameters(), model_reloaded.named_parameters()
        ):
            assert name == name_exp
            assert torch.allclose(param, param_exp, rtol=1e-3, atol=1e-4)

    def test_export_model__wrapped_model_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = fastervit.create_model("faster_vit_0_224", pretrained=False)
        wrapped_model = FasterViTPackage.get_model_wrapper(model=model)

        FasterViTPackage.export_model(model=wrapped_model, out=out)
        assert out.exists()

    def test_export_model__unsupported_model(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = DummyCustomModel()

        with pytest.raises(ValueError, match="FasterViTPackage only supports"):
            FasterViTPackage.export_model(model=model, out=out)

        with pytest.raises(ValueError, match="FasterViTPackage only supports"):
            FasterViTPackage.export_model(model=model.get_model(), out=out)
