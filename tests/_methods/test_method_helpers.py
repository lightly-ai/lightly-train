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
from lightly_train._methods.distillation.distillation import Distillation
from lightly_train._methods.method import Method
from lightly_train._methods.simclr.simclr import SimCLR

from .. import helpers


@pytest.mark.parametrize(
    "method, expected",
    [
        ("densecl", DenseCL),
        ("dino", DINO),
        ("simclr", SimCLR),
        ("distillation", Distillation),
        (helpers.get_method(), SimCLR),
    ],
)
def test_get_method_cls(method: str, expected: type[Method]) -> None:
    assert method_helpers.get_method_cls(method=method) == expected


def test_list_methods_private() -> None:
    assert method_helpers._list_methods() == [
        "densecl",
        "dino",
        "distillation",
        "simclr",
    ]


def test_list_methods_public() -> None:
    assert method_helpers.list_methods() == [
        "dino",
        "distillation",
        "simclr",
    ]
