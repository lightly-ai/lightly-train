#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path

from pytest_mock import MockerFixture

from lightly_train._commands import common_helpers
from lightly_train._data import cache


def test_get_model_cache_dir__default() -> None:
    expected = Path.home() / ".cache" / "lightly-train" / "models"
    assert cache.get_model_cache_dir() == expected


def test_get_model_cache_dir__custom(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_MODEL_CACHE_DIR": str(tmp_path)})
    assert cache.get_model_cache_dir() == tmp_path


def test_get_data_cache_dir() -> None:
    expected = Path.home() / ".cache" / "lightly-train" / "data"
    assert cache.get_data_cache_dir() == expected


def test_get_data_cache_dir__custom(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_DATA_CACHE_DIR": str(tmp_path)})
    assert cache.get_data_cache_dir() == tmp_path


def test_get_data_cache_dir__tmp_dir_propagation(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Ensure that the temporary directory is propagated to the data cache directory.
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TMP_DIR": str(tmp_path)})
    assert cache.get_data_cache_dir() == tmp_path
    assert cache.get_data_cache_dir() == common_helpers.get_tmp_dir()
