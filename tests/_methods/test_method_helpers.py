#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._methods import method_helpers
from lightly_train._methods.densecl.densecl import DenseCL
from lightly_train._methods.dino.dino import DINO
from lightly_train._methods.dinov2.dinov2 import DINOv2
from lightly_train._methods.distillation.distillation import Distillation
from lightly_train._methods.distillationv2.distillationv2 import DistillationV2
from lightly_train._methods.method import Method
from lightly_train._methods.simclr.simclr import SimCLR

from .. import helpers
from ..helpers import DummyCustomModel


@pytest.mark.parametrize(
    "method, expected",
    [
        ("densecl", DenseCL),
        ("dino", DINO),
        ("dinov2", DINOv2),
        ("simclr", SimCLR),
        ("distillationv1", Distillation),
        ("distillationv2", DistillationV2),
        (helpers.get_method(wrapped_model=DummyCustomModel()), SimCLR),
    ],
)
def test_get_method_cls(method: str, expected: type[Method]) -> None:
    assert method_helpers.get_method_cls(method=method) == expected


def test_list_methods_private() -> None:
    assert method_helpers._list_methods() == [
        "densecl",
        "dino",
        "dinov2",
        "distillation",
        "distillationv1",
        "distillationv2",
        "simclr",
    ]


def test_list_methods_public() -> None:
    assert method_helpers.list_methods() == [
        "dino",
        "dinov2",
        "distillation",
        "distillationv1",
        "distillationv2",
        "simclr",
    ]
