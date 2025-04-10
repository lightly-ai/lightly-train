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
from lightly_train._models.package import Package
from lightly_train._models.timm.timm_package import TIMM_PACKAGE
from tests.helpers import DummyCustomModel


@pytest.mark.parametrize("package", [CUSTOM_PACKAGE, TIMM_PACKAGE])
def test_list_packages(package: Package) -> None:
    assert package in package_helpers.list_packages()


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
    from rfdetr.models.lwdetr import LWDETR

    model = package_helpers.get_model("rfdetr/rf-detr-base")
    assert isinstance(model, LWDETR)


def test_get_model__torchvision() -> None:
    model = package_helpers.get_model("torchvision/resnet18")
    assert isinstance(model, ResNet)


def test_get_model__timm() -> None:
    pytest.importorskip("timm")
    from timm.models.resnet import ResNet

    model = package_helpers.get_model("timm/resnet18")
    assert isinstance(model, ResNet)


def test_get_model__super_gradients() -> None:
    pytest.importorskip("super_gradients")
    from super_gradients.training.models import (
        YoloNAS_S,
    )

    model = package_helpers.get_model("super_gradients/yolo_nas_s")
    assert isinstance(model, YoloNAS_S)


def test_get_model__ultralytics() -> None:
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    model = package_helpers.get_model("ultralytics/yolov8s.yaml")
    assert isinstance(model, YOLO)


def test_get_model__custom() -> None:
    model = package_helpers.get_model(model=DummyCustomModel())
    assert CUSTOM_PACKAGE.is_supported_model(model)


def test_get_feature_extractor__timm() -> None:
    pytest.importorskip("timm")
    model = package_helpers.get_model("timm/resnet18")
    feature_extractor = package_helpers.get_feature_extractor(model=model)

    x = torch.rand(1, 3, 64, 64)
    y_model = model(x)
    y_extractor = model.forward_head(feature_extractor.forward_features(x)["features"])
    torch.testing.assert_close(y_model, y_extractor)
