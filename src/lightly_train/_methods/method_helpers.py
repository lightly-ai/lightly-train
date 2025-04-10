#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._methods.densecl.densecl import DenseCL
from lightly_train._methods.dino.dino import DINO
from lightly_train._methods.distillation.distillation import Distillation
from lightly_train._methods.method import Method
from lightly_train._methods.simclr.simclr import SimCLR

HIDDEN_METHODS = {DenseCL.__name__.lower()}


def list_methods() -> list[str]:
    """Lists all available self-supervised learning methods.

    See the documentation for more information: https://docs.lightly.ai/train/stable/methods/
    """
    return sorted(list(set(_method_name_to_cls().keys()) - HIDDEN_METHODS))


def _list_methods() -> list[str]:
    """Lists all available self-supervised learning methods. Including the hidden ones.

    See the documentation for more information: https://docs.lightly.ai/train/stable/methods/
    """
    return sorted(_method_name_to_cls().keys())


def get_method_cls(method: str | Method) -> type[Method]:
    if isinstance(method, Method):
        return method.__class__
    method_cls = _method_name_to_cls().get(method)
    if method_cls is not None:
        return method_cls
    else:
        raise ValueError(
            f"Method '{method}' is unknown. Available methods are: {_list_methods()}"
        )


def _method_name_to_cls() -> dict[str, type[Method]]:
    return {
        m.__name__.lower(): m
        for m in [
            DenseCL,
            DINO,
            Distillation,
            SimCLR,
        ]
    }
