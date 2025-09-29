#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re
from pathlib import Path
from typing import Generator

import pytest
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
    yield cache_dir
