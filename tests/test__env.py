#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightly_train._env import EnvVar


class TestEnvVar:
    def test_name(self) -> None:
        env_var = EnvVar(name="LIGHTLY_TRAIN_TEST_ENV_VAR", _default=42, _type=int)
        assert env_var.name == "LIGHTLY_TRAIN_TEST_ENV_VAR"

    @pytest.mark.parametrize(
        "env_value, default, type_, convert_empty_str_to_default, expected",
        [
            (None, 42, int, True, 42),
            ("100", 42, int, True, 100),
            ("", 42, int, True, 42),
            ("", "42", str, True, "42"),
            ("", "42", str, False, ""),
        ],
    )
    def test_value(
        self,
        mocker: MockerFixture,
        env_value: str | None,
        default: int,
        type_: Any,
        convert_empty_str_to_default: bool,
        expected: int,
    ) -> None:
        env_var = EnvVar(
            name="LIGHTLY_TRAIN_TEST_ENV_VAR",
            _default=default,
            _type=type_,
            convert_empty_str_to_default=convert_empty_str_to_default,
        )
        if env_value is not None:
            mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TEST_ENV_VAR": env_value})
        assert env_var.value == expected

    def test_value__dynamic_default(self, mocker: MockerFixture) -> None:
        def default_value() -> int:
            return 42

        env_var = EnvVar(
            name="LIGHTLY_TRAIN_TEST_ENV_VAR",
            _default=default_value,
            _type=int,
        )
        assert env_var.value == 42
        mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TEST_ENV_VAR": "100"})
        assert env_var.value == 100

    def test_raw_value(self, mocker: MockerFixture) -> None:
        env_var = EnvVar(name="LIGHTLY_TRAIN_TEST_ENV_VAR", _default=42, _type=int)
        mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TEST_ENV_VAR": "100"})
        assert env_var.raw_value == "100"

    def test_raw_value__default(self) -> None:
        env_var = EnvVar(name="LIGHTLY_TRAIN_TEST_ENV_VAR", _default=42, _type=int)
        assert env_var.raw_value == "42"

    def test_raw_value__default_none(self) -> None:
        env_var = EnvVar(name="LIGHTLY_TRAIN_TEST_ENV_VAR", _default=None, _type=int)
        assert env_var.raw_value is None
