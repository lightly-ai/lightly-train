#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from importlib import util as importlib_util
from pathlib import Path

import pytest
import torch

from lightly_train._models.rfdetr.rfdetr import RFDETRModelWrapper
from lightly_train._models.rfdetr.rfdetr_package import RFDETRPackage

from ...helpers import DummyCustomModel

if importlib_util.find_spec("rfdetr") is None:
    pytest.skip("rfdetr is not installed", allow_module_level=True)

from rfdetr.detr import RFDETR, RFDETRBase
from rfdetr.models.backbone.dinov2 import WindowedDinov2WithRegistersBackbone


class TestRFDETRPackage:
    @pytest.mark.parametrize(
        "model_name, supported",
        [
            ("rfdetr/rf-detr-base", True),
            ("rfdetr/rf-detr-small", False),  # No pretrained checkpoint available.
            ("rfdetr/rf-detr-base-2", True),
            ("rfdetr/rf-detr-small-2", False),  # No pretrained checkpoint available.
            ("rfdetr/rf-detr-large", True),
            ("rfdetr/rf-detr-large-2", False),  # No pretrained checkpoint available.
        ],
    )
    def test_list_model_names(self, model_name: str, supported: bool) -> None:
        model_names = RFDETRPackage.list_model_names()
        assert (model_name in model_names) is supported

    def test_is_supported_model__model_true(self) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        assert RFDETRPackage.is_supported_model(model=model)

        wrapped_model = RFDETRModelWrapper(model=model)
        assert RFDETRPackage.is_supported_model(model=wrapped_model)

    def test_is_supported_model__wrapped_model_false(self) -> None:
        model = DummyCustomModel()
        assert not RFDETRPackage.is_supported_model(model=model)

    def test_is_supported_model__model_false(self) -> None:
        model = DummyCustomModel().get_model()
        assert not RFDETRPackage.is_supported_model(model=model)

    @pytest.mark.parametrize(
        "model_name",
        ["rf-detr-base", "rf-detr-base-2", "rf-detr-large"],
    )
    def test_get_model(self, model_name: str) -> None:
        model = RFDETRPackage.get_model(model_name=model_name)
        assert isinstance(model, RFDETR)

    def test_get_model_wrapper(self) -> None:
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        fe = RFDETRPackage.get_model_wrapper(model=model)
        assert isinstance(fe, RFDETRModelWrapper)

    def test_export_model__model(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = RFDETRBase()  # type: ignore[no-untyped-call]

        RFDETRPackage.export_model(model=model, out=out)
        model_exported = RFDETRBase(pretrain_weights=out.as_posix())  # type: ignore[no-untyped-call]

        lwdetr_model = model.model.model
        lwdetr_model_exported = model_exported.model.model

        # Check that parameters are the same.
        assert len(list(lwdetr_model.parameters())) == len(
            list(lwdetr_model_exported.parameters())
        )
        for (name, param), (name_exp, param_exp) in zip(
            lwdetr_model.named_parameters(), lwdetr_model_exported.named_parameters()
        ):
            assert name == name_exp
            assert param.dtype == param_exp.dtype
            assert param.requires_grad == param_exp.requires_grad
            assert torch.allclose(param, param_exp, rtol=1e-3, atol=1e-4)

        # Check module states. The pretrained DINOv2 backbone is frozen while other modules are in training mode.
        visited = set()
        for (name, module), (name_exp, module_exp) in zip(
            lwdetr_model.named_modules(), lwdetr_model_exported.named_modules()
        ):
            if name in visited:
                continue

            assert name == name_exp
            if isinstance(
                module, WindowedDinov2WithRegistersBackbone
            ):  # Check the state for all modules in DINOv2 backbone
                for (child_name, child_module), (
                    child_name_exp,
                    child_module_exp,
                ) in zip(module.named_modules(), module_exp.named_modules()):
                    assert child_name == child_name_exp
                    assert not child_module.training
                    assert not child_module_exp.training

                    visited.add(f"{name}.{child_name}")
            else:
                assert module.training
                assert module_exp.training

    def test_export_model__wrapped_model(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = RFDETRBase()  # type: ignore[no-untyped-call]
        wrapped_model = RFDETRModelWrapper(model=model)
        RFDETRPackage.export_model(model=wrapped_model, out=out)
        RFDETRBase(pretrain_weights=out.as_posix())  # type: ignore[no-untyped-call]

    def test_export_model__unsupported_model(self, tmp_path: Path) -> None:
        out = tmp_path / "model.pt"
        model = DummyCustomModel()
        with pytest.raises(
            ValueError, match="Model must be of type 'RFDETR' or 'RFDETRModelWrapper'"
        ):
            RFDETRPackage.export_model(model=model, out=out)
