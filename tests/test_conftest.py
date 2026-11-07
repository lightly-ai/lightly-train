#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import sys

import pytest
from pytest_mock import MockerFixture

from tests.conftest import _missing_requirement, _required_gpus


@pytest.mark.parametrize(
    "version, expected",
    [
        ((3, 8, 0), "python>=3.9"),
        ((3, 9, 0), None),
        ((3, 10, 0), None),
    ],
)
def test__missing_requirement__python_geq(
    mocker: MockerFixture, version: tuple[int, int, int], expected: str | None
) -> None:
    mocker.patch.object(sys, "version_info", version)
    assert _missing_requirement("python>=3.9") == expected


@pytest.mark.parametrize(
    "version, expected",
    [
        ((3, 8, 0), None),
        ((3, 9, 0), "python<3.9"),
    ],
)
def test__missing_requirement__python_lt(
    mocker: MockerFixture, version: tuple[int, int, int], expected: str | None
) -> None:
    mocker.patch.object(sys, "version_info", version)
    assert _missing_requirement("python<3.9") == expected


@pytest.mark.parametrize(
    "requirement",
    [
        "Python>=3.9",
        "python>=3.9.1",
        "python==3.9",
        "python>=3.9; sys_platform=='win32'",
    ],
)
def test__missing_requirement__python_invalid(requirement: str) -> None:
    with pytest.raises(ValueError, match="Unsupported python requirement"):
        _missing_requirement(requirement)


def test__missing_requirement__package_with_python_prefix() -> None:
    assert (
        _missing_requirement("python-not-installed-package")
        == "python-not-installed-package"
    )


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        ((), {}, 1),
        ((2,), {}, 2),
        ((), {"n": 2}, 2),
    ],
)
def test__required_gpus(
    args: tuple[int, ...], kwargs: dict[str, int], expected: int
) -> None:
    assert _required_gpus(args, kwargs) == expected


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((0,), {}),
        ((True,), {}),
        ((), {"n": "2"}),
        ((1, 2), {}),
        ((1,), {"n": 1}),
        ((), {"count": 2}),
    ],
)
def test__required_gpus__invalid(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="gpu"):
        _required_gpus(args, kwargs)
