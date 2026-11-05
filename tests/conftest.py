#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Mapping

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache
from pytest import FixtureRequest, TempPathFactory
from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)  # Apply to all tests
def lightly_train_cache_dir(
    tmp_path_factory: TempPathFactory,
    request: FixtureRequest,
    mocker: MockerFixture,
) -> Generator[Path, None, None]:
    """Set LIGHTLY_TRAIN_CACHE_DIR to a unique directory for each test.

    This ensures that tests do not share cache files between each other. By default
    LightlyTrain uses ~/.cache/lightly-train which is the same for all tests and can
    lead to hard-to-debug issues when tests interfere with each other.
    """
    name = request.node.name
    # From: https://github.com/pytest-dev/pytest/blob/9913cedb51a39da580d3ef3aff8cff006c3e7fc6/src/_pytest/tmpdir.py#L247-L249
    name = re.sub(r"[\W]", "_", name)
    MAXVAL = 100
    name = name[:MAXVAL]
    # Use tmp_path_factory instead of tmp_path because tmp_path is oftentimes also used
    # inside the actual test function. We don't want to use tmp_path for the cache dir
    # because then tmp_path is not empty anymore for the test function which might be
    # unexpected.
    cache_dir = tmp_path_factory.mktemp(f"{name}_lightly_train_cache")
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_CACHE_DIR": str(cache_dir)})
    try:
        yield cache_dir
    finally:
        # Delete the cache dir after the test. By default, pytest only deletes
        # directories created via tmp_path_factory at the end of the whole test
        # session.
        shutil.rmtree(cache_dir, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "long_running_test: marks tests as long running. Excluded from "
        "test-ci-minimal/test-ci-maximal via '-m \"not long_running_test\"', but "
        "still run by 'make test' and by the unfiltered nightly CI in "
        "lightly-train-private.",
    )
    config.addinivalue_line(
        "markers",
        "skip_windows(reason): skip on Windows. Only for behavior that differs on "
        "Windows by design, not for slow or failing tests.",
    )
    config.addinivalue_line(
        "markers",
        "requires(*requirements, reason=None): skip unless all requirements are "
        "met. Takes requirement strings or RequirementCache constants, e.g. "
        'requires("onnx", TORCH_DIM_HINTS_AVAILABLE, "python>=3.10", reason="...").',
    )
    config.addinivalue_line(
        "markers",
        "gpu(n=1): skip unless at least n CUDA devices are available.",
    )


@lru_cache(maxsize=None)
def _requirement_cache(requirement: str) -> RequirementCache:
    return RequirementCache(requirement)


_PYTHON_REQUIREMENT = re.compile(r"python(>=|<)(\d+)\.(\d+)")


def _missing_requirement(requirement: str | RequirementCache) -> str | None:
    """Return the requirement name if it is not met, otherwise None."""
    if isinstance(requirement, str):
        match = _PYTHON_REQUIREMENT.fullmatch(requirement)
        if match is not None:
            operator, major, minor = match.groups()
            version = (int(major), int(minor))
            if operator == ">=":
                met = sys.version_info[:2] >= version
            else:
                met = sys.version_info[:2] < version
            return None if met else requirement
        if re.match(r"python\s*[<>=!~]", requirement, re.IGNORECASE):
            raise ValueError(
                f"Unsupported python requirement {requirement!r}. Use 'python>=X.Y' "
                "or 'python<X.Y'."
            )
        requirement = _requirement_cache(requirement)
    if requirement:
        return None
    return requirement.requirement or requirement.module


@lru_cache(maxsize=None)
def _cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _required_gpus(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int:
    """Return n from gpu(n) or gpu(n=n). Defaults to 1."""
    if len(args) > 1 or set(kwargs) - {"n"} or (args and kwargs):
        raise ValueError("gpu takes a single argument n.")
    n = args[0] if args else kwargs.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"gpu needs n to be a positive int, got {n!r}.")
    return n


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        for marker in item.iter_markers("skip_windows"):
            reason = marker.kwargs.get("reason")
            if not reason:
                raise pytest.UsageError(
                    f"{item.nodeid}: skip_windows requires a reason."
                )
            if sys.platform.startswith("win"):
                item.add_marker(pytest.mark.skip(reason=reason))
        for marker in item.iter_markers("requires"):
            if not marker.args:
                raise pytest.UsageError(
                    f"{item.nodeid}: requires needs at least one requirement."
                )
            try:
                missing = [
                    name
                    for name in (_missing_requirement(r) for r in marker.args)
                    if name is not None
                ]
            except ValueError as e:
                raise pytest.UsageError(f"{item.nodeid}: {e}") from e
            if missing:
                reason = marker.kwargs.get("reason")
                message = f"Requires {', '.join(missing)}"
                message = f"{message}: {reason}" if reason else f"{message}."
                item.add_marker(pytest.mark.skip(reason=message))
        for marker in item.iter_markers("gpu"):
            try:
                n = _required_gpus(marker.args, marker.kwargs)
            except ValueError as e:
                raise pytest.UsageError(f"{item.nodeid}: {e}") from e
            if _cuda_device_count() < n:
                item.add_marker(
                    pytest.mark.skip(reason=f"Requires {n} CUDA device(s).")
                )


@pytest.fixture(autouse=True)  # Apply to all tests
def set_test_env_variables(
    mocker: MockerFixture,
) -> None:
    mocker.patch.dict(
        os.environ,
        {
            "LIGHTLY_TRAIN_EVENTS_DISABLED": "1",
            "LIGHTLY_TRAIN_POSTHOG_KEY": "",
        },
    )
