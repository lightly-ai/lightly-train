#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pytest
import torch
from torchvision.models import ResNet

from lightly_train._models import package_helpers
from lightly_train._models.custom.custom_package import CUSTOM_PACKAGE
from lightly_train._models.package import BasePackage
from lightly_train._models.timm.timm_package import TIMM_PACKAGE
from lightly_train.errors import UnknownModelError

from ..helpers import DummyCustomModel


@pytest.mark.parametrize("package", [CUSTOM_PACKAGE, TIMM_PACKAGE])
def test_list_packages(package: BasePackage) -> None:
    assert package in package_helpers.list_base_packages()


def test_get_package() -> None:
    assert package_helpers.get_package("timm") == TIMM_PACKAGE

    with pytest.raises(ValueError):
        assert package_helpers.get_package("other")


@pytest.mark.parametrize(
    "package_name, model_name",
    [
        ("rfdetr", "rfdetr/rf-detr-base"),
        ("super_gradients", "super_gradients/yolo_nas_s"),
        ("timm", "timm/resnet18"),
        ("torchvision", "torchvision/resnet18"),
        ("ultralytics", "ultralytics/yolov8s.yaml"),
    ],
)
def test_list_model_names(package_name: str, model_name: str) -> None:
    pytest.importorskip(package_name)
    assert model_name in package_helpers.list_model_names()


def test_get_model__rfdetr() -> None:
    pytest.importorskip("rfdetr")
    from rfdetr.detr import RFDETR

    model = package_helpers.get_wrapped_model("rfdetr/rf-detr-base")
    assert isinstance(model.get_model(), RFDETR)


def test_get_model__torchvision() -> None:
    model = package_helpers.get_wrapped_model("torchvision/resnet18")
    assert isinstance(model.get_model(), ResNet)


def test_get_model__timm() -> None:
    pytest.importorskip("timm")
    from timm.models.resnet import ResNet

    model = package_helpers.get_wrapped_model("timm/resnet18")
    assert isinstance(model.get_model(), ResNet)


def test_get_model__super_gradients() -> None:
    pytest.importorskip("super_gradients")
    from super_gradients.training.models import (
        YoloNAS_S,
    )

    model = package_helpers.get_wrapped_model("super_gradients/yolo_nas_s")
    assert isinstance(model.get_model(), YoloNAS_S)


def test_get_model__ultralytics() -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    model = package_helpers.get_wrapped_model("ultralytics/yolov8s.yaml")
    assert isinstance(model.get_model(), YOLO)


def test_get_model_wrapper__timm() -> None:
    pytest.importorskip("timm")
    wrapped_model = package_helpers.get_wrapped_model("timm/resnet18")
    model = wrapped_model.get_model()

    x = torch.rand(1, 3, 64, 64)
    y_model = model(x)
    y_extractor = model.forward_head(wrapped_model.forward_features(x)["features"])
    torch.testing.assert_close(y_model, y_extractor)


def test_get_package_from_model__custom() -> None:
    assert (
        package_helpers.get_package_from_model(  # type: ignore[comparison-overlap]
            model=DummyCustomModel(), include_custom=True, fallback_custom=False
        )
        == CUSTOM_PACKAGE
    )


def test_get_package_from_model__custom_invalid() -> None:
    class InvalidCustomModelWrapper:
        def get_model(self) -> None:
            pass

        def forward_features(self) -> None:
            pass

    with pytest.raises(
        UnknownModelError,
        match=(
            r"Unknown model: 'InvalidCustomModelWrapper'. If you are "
            r"implementing a custom model wrapper, please make sure the wrapper class "
            r"inherits from torch.nn.Module and implements all required methods.\n"
            r" - Inherits from torch.nn.Module: False\n"
            r" - Missing methods: \['feature_dim', 'forward_pool'\]"
        ),
    ):
        package_helpers.get_package_from_model(  # type: ignore[call-overload]
            model=InvalidCustomModelWrapper(),
            include_custom=True,
            fallback_custom=False,
        )
