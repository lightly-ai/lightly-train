#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import pytest

from lightly_train._configs.config import PydanticConfig


class _Config(PydanticConfig):
    a: Any


class _Default(PydanticConfig):
    a: str = "auto"


class TestPydanticConfig:
    @pytest.mark.parametrize(
        "config, expected",
        [
            # Config
            (_Config(a=""), False),
            (_Config(a="test"), False),
            (_Config(a="auto"), True),
            # Dict
            (_Config(a={"a": ""}), False),
            (_Config(a={"a": "test"}), False),
            (_Config(a={"a": "auto"}), True),
            # Config in config
            (_Config(a=_Config(a="")), False),
            (_Config(a=_Config(a="auto")), True),
            # Config in dict
            (_Config(a={"a": _Config(a="")}), False),
            (_Config(a={"a": _Config(a="auto")}), True),
            # Dict in config
            (_Config(a=_Config(a={"a": ""})), False),
            (_Config(a=_Config(a={"a": "auto"})), True),
            # Dict in dict
            (_Config(a={"a": {"b": "auto"}}), True),
            # Default
            (_Default(), True),
            (_Default(a=""), False),
            (_Config(a=_Default()), True),
            (_Config(a=_Default(a="")), False),
        ],
    )
    def test_has_auto(self, config: PydanticConfig, expected: bool) -> None:
        assert config.has_auto() == expected
