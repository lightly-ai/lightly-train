#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from itertools import chain
from typing import Any, Literal, overload

from torch.nn import Module

from lightly_train._models import model_wrapper
from lightly_train._models.custom.custom_package import CUSTOM_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import BasePackage, Package
from lightly_train._models.rfdetr.rfdetr_package import RFDETR_PACKAGE
from lightly_train._models.super_gradients.super_gradients_package import (
    SUPER_GRADIENTS_PACKAGE,
)
from lightly_train._models.timm.timm_package import TIMM_PACKAGE
from lightly_train._models.torchvision.torchvision_package import TORCHVISION_PACKAGE
from lightly_train._models.ultralytics.ultralytics_package import ULTRALYTICS_PACKAGE
from lightly_train.errors import UnknownModelError


def list_base_packages() -> list[BasePackage]:
    """Lists all supported packages."""
    return [
        RFDETR_PACKAGE,
        SUPER_GRADIENTS_PACKAGE,
        TIMM_PACKAGE,
        TORCHVISION_PACKAGE,
        ULTRALYTICS_PACKAGE,
        DINOV2_VIT_PACKAGE,
        DINOV3_PACKAGE,
        # Custom package must be at end of list because we first want to check if a
        # model is part of one of the other packages. Custom is the last resort.
        CUSTOM_PACKAGE,
    ]


def list_packages() -> list[Package]:
    """Lists all supported framework packages."""
    return [package for package in list_base_packages() if isinstance(package, Package)]


def get_package(package_name: str) -> Package:
    """Get a package by name."""
    packages = {p.name: p for p in list_packages()}
    try:
        return packages[package_name]
    except KeyError:
        raise ValueError(
            f"Unknown package name: '{package_name}'. Supported packages are "
            f"{list(packages)}."
        )


def list_model_names() -> list[str]:
    """Lists all models in ``<package_name>/<model_name>`` format.

    See the documentation for more information: https://docs.lightly.ai/train/stable/models/
    """
    return sorted(chain.from_iterable(p.list_model_names() for p in list_packages()))


def get_wrapped_model(
    model: str | Module | ModelWrapper, model_args: dict[str, Any] | None = None
) -> ModelWrapper:
    """Returns a wrapped model instance given a model name or instance."""
    if isinstance(model, ModelWrapper):
        return model

    package: Package
    if isinstance(model, str):
        package_name, model_name = parse_model_name(model)
        package = get_package(package_name)
        model = package.get_model(model_name, model_args=model_args)
    else:
        package = get_package_from_model(
            model, include_custom=False, fallback_custom=False
        )
    return package.get_model_wrapper(model)


@overload
def get_package_from_model(
    model: Module | ModelWrapper, include_custom: bool, fallback_custom: Literal[True]
) -> BasePackage: ...


@overload
def get_package_from_model(
    model: Module | ModelWrapper, include_custom: bool, fallback_custom: Literal[False]
) -> Package: ...


def get_package_from_model(
    model: Module | ModelWrapper,
    include_custom: bool,
    fallback_custom: bool,
) -> BasePackage | Package:
    """Returns the package of the model."""
    packages = list_base_packages() if include_custom else list_packages()
    for package in packages:
        if package.is_supported_model(model):
            return package

    if not fallback_custom:
        is_torch_module = isinstance(model, Module)
        missing_attrs = model_wrapper.missing_model_wrapper_attrs(
            model, exclude_module_attrs=True
        )
        raise UnknownModelError(
            f"Unknown model: '{model.__class__.__name__}'. If you are implementing a "
            "custom model wrapper, please make sure the wrapper class inherits from "
            "torch.nn.Module and implements all required methods.\n"
            f" - Inherits from torch.nn.Module: {is_torch_module}\n"
            f" - Missing methods: {missing_attrs}\n"
            "For more information, please refer to the documentation: https://docs.lightly.ai/train/stable/models/custom_models.html"
        )
    else:
        return CUSTOM_PACKAGE


def parse_model_name(model: str) -> tuple[str, str]:
    parts = model.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Model name has incorrect format. Should be 'package/model' but is "
            f"'{model}'"
        )
    package_name = parts[0]
    model_name = parts[1]
    if package_name == "dinov2_vit":  # For backwards compatibility.
        package_name = "dinov2"
    return package_name, model_name
