#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from lightly_train._configs import validate
from lightly_train._configs.config import PydanticConfig
from lightly_train.errors import (
    ConfigValidationError,
    UnresolvedAutoError,
)


class _SimpleConfig(PydanticConfig):
    a: int


class _OptionalConfig(PydanticConfig):
    a: int
    b: str = "default"


class _UnionConfig(PydanticConfig):
    a: int | str | Literal["auto"] | Path


class _NestedConfig(PydanticConfig):
    class A(PydanticConfig):
        b: int

    a: A


@pytest.mark.parametrize(
    "cfg, obj, expected",
    [
        (_SimpleConfig, {"a": 1}, _SimpleConfig(a=1)),
        (_OptionalConfig, {"a": 1}, _OptionalConfig(a=1, b="default")),
        (_OptionalConfig, {"a": 1, "b": "custom"}, _OptionalConfig(a=1, b="custom")),
        (_UnionConfig, {"a": 1}, _UnionConfig(a=1)),
        (_UnionConfig, {"a": "auto"}, _UnionConfig(a="auto")),
        (_UnionConfig, {"a": Path("path")}, _UnionConfig(a=Path("path"))),
        (_NestedConfig, {"a": {"b": 1}}, _NestedConfig(a=_NestedConfig.A(b=1))),
        (
            _NestedConfig,
            {"a": _NestedConfig.A(b=1)},
            _NestedConfig(a=_NestedConfig.A(b=1)),
        ),
    ],
)
def test_pydantic_model_validate(
    cfg: type[PydanticConfig], obj: dict[str, Any], expected: PydanticConfig
) -> None:
    validated = validate.pydantic_model_validate(model=cfg, obj=obj)
    assert validated == expected


@pytest.mark.parametrize(
    "cfg, obj, errors",
    [
        (_SimpleConfig, {}, ["Missing key: 'a'"]),
        (_SimpleConfig, {"a": 1, "b": 2}, ["Unknown key: 'b'"]),
        (
            _SimpleConfig,
            {"a": "1"},
            [
                "Invalid type for key 'a': Input should be a valid integer but got '1' with type 'str'"
            ],
        ),
        (_OptionalConfig, {}, ["Missing key: 'a'"]),
        (
            _UnionConfig,
            {"a": 0.1},
            [
                "Invalid type for key 'a.int': Input should be a valid integer but got 0.1 with type 'float'",
                "Invalid type for key 'a.str': Input should be a valid string but got 0.1 with type 'float'",
                "Invalid type for key 'a.literal['auto']': Input should be 'auto' but got 0.1 with type 'float'",
                "Invalid type for key 'a.Path': Input should be an instance of Path but got 0.1 with type 'float'",
            ],
        ),
        (_NestedConfig, {"a": {}}, ["Missing key: 'a.b'"]),
        (_NestedConfig, {"a": {"b": 1, "c": 1}}, ["Unknown key: 'a.c'"]),
        (
            _NestedConfig,
            {"a": 1},
            [
                "Invalid type for key 'a': Input should be a valid dictionary or instance of A but got 1 with type 'int'"
            ],
        ),
        (
            _NestedConfig,
            {"a": {"b": "1"}},
            [
                "Invalid type for key 'a.b': Input should be a valid integer but got '1' with type 'str'"
            ],
        ),
    ],
)
def test_pydantic_model_validate__error(
    cfg: type[PydanticConfig], obj: dict[str, Any], errors: list[str]
) -> None:
    with pytest.raises(
        ConfigValidationError, match=f"Found {len(errors)} errors"
    ) as ex_info:
        validate.pydantic_model_validate(model=cfg, obj=obj)
    for error in errors:
        assert error in str(ex_info.value)


@pytest.mark.parametrize(
    "config, other, expected",
    [
        (_SimpleConfig(a=1), {}, _SimpleConfig(a=1)),
        (_SimpleConfig(a=1), {"a": 2}, _SimpleConfig(a=2)),
        (_OptionalConfig(a=1), {"a": 2}, _OptionalConfig(a=2, b="default")),
        (_OptionalConfig(a=1), {"a": 2, "b": "other"}, _OptionalConfig(a=2, b="other")),
        (
            _NestedConfig(a=_NestedConfig.A(b=1)),
            {"a": {"b": 2}},
            _NestedConfig(a=_NestedConfig.A(b=2)),
        ),
    ],
)
def test_pydantic_model_merge(
    config: PydanticConfig, other: dict[str, Any], expected: PydanticConfig
) -> None:
    assert validate.pydantic_model_merge(config, other) == expected


def test_no_auto() -> None:
    # Type out to make sure mypy doesn't complain.
    out: int = validate.no_auto(1)
    assert out == 1
    with pytest.raises(UnresolvedAutoError):
        validate.no_auto("auto")
