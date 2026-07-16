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
from torch import Tensor
from torch.nn import Conv2d, Module

from lightly_train._models import package_helpers
from lightly_train._models.radio.radio import RadioModelWrapper
from lightly_train._models.radio.radio_package import (
    MODEL_NAMES,
    RADIO_TORCH_HUB_REVISION,
    RadioPackage,
)


class FakeRadioModel(Module):
    embed_dim = 8
    min_resolution_step = 16

    def __init__(self) -> None:
        super().__init__()
        self.projection = Conv2d(3, self.embed_dim, kernel_size=1)
        self.feature_formats: list[str] = []

    def forward(self, x: Tensor, feature_fmt: str) -> tuple[Tensor, Tensor]:
        self.feature_formats.append(feature_fmt)
        features = self.projection(x)
        return features.mean(dim=(-2, -1)), features


class TestRadioPackage:
    def test_list_model_names(self) -> None:
        assert RadioPackage.list_model_names() == [
            f"radio/{model_name}" for model_name in MODEL_NAMES
        ]
        assert package_helpers.get_package("radio") is not None
        assert "radio/c-radio_v4-h" in package_helpers.list_model_names()

    def test_get_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_load(*args: object, **kwargs: object) -> FakeRadioModel:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeRadioModel()

        monkeypatch.setattr(torch.hub, "load", fake_load)

        model = RadioPackage.get_model("c-radio_v4-h")

        assert isinstance(model, FakeRadioModel)
        assert RadioPackage.is_supported_model(model)
        assert captured["args"] == (
            f"NVlabs/RADIO:{RADIO_TORCH_HUB_REVISION}",
            "radio_model",
        )
        assert captured["kwargs"] == {
            "version": "c-radio_v4-h",
            "progress": True,
            "skip_validation": True,
        }

    def test_get_model__hub_ref_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_load(*args: object, **kwargs: object) -> FakeRadioModel:
            captured["args"] = args
            return FakeRadioModel()

        monkeypatch.setattr(torch.hub, "load", fake_load)

        RadioPackage.get_model("radio_v1", model_args={"hub_ref": "test-ref"})

        assert captured["args"] == ("NVlabs/RADIO:test-ref", "radio_model")

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"model_name": "unknown"}, "Unknown RADIO model"),
            ({"model_name": "radio_v1", "num_input_channels": 1}, "3 input"),
            ({"model_name": "radio_v1", "load_weights": False}, "load_weights=False"),
            (
                {"model_name": "radio_v1", "model_args": {"invalid": True}},
                "Unsupported RADIO model_args",
            ),
        ],
    )
    def test_get_model__invalid_args(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            RadioPackage.get_model(**kwargs)  # type: ignore[arg-type]

    def test_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeRadioModel())
        model = RadioPackage.get_model("radio_v1")
        wrapper = RadioPackage.get_model_wrapper(model)

        output = wrapper.forward_features(torch.rand(2, 3, 32, 48))
        pooled = wrapper.forward_pool(output)

        assert isinstance(wrapper, RadioModelWrapper)
        assert wrapper.feature_dim() == 8
        assert output["features"].shape == (2, 8, 32, 48)
        assert pooled["pooled_features"].shape == (2, 8, 1, 1)
        assert model.feature_formats == ["NCHW"]  # type: ignore[attr-defined]

    def test_wrapper__invalid_resolution(self) -> None:
        wrapper = RadioModelWrapper(FakeRadioModel())

        with pytest.raises(ValueError, match="min_resolution_step=16"):
            wrapper.forward_features(torch.rand(1, 3, 31, 32))

    def test_export_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeRadioModel())
        model = RadioPackage.get_model("radio_v1")
        out = tmp_path / "radio.pt"

        RadioPackage.export_model(RadioPackage.get_model_wrapper(model), out)
        exported = torch.load(out, weights_only=True)

        assert exported.keys() == model.state_dict().keys()

    def test_get_model_wrapper__unsupported_model(self) -> None:
        with pytest.raises(ValueError, match="RadioPackage cannot create"):
            RadioPackage.get_model_wrapper(Conv2d(3, 8, kernel_size=1))
