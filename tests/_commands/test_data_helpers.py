#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import Union

import yaml
from pydantic import AliasChoices, BaseModel, Field

DATA_HELPERS_PATH = (
    Path(__file__).parents[2]
    / "src"
    / "lightly_train"
    / "_commands"
    / "data_helpers.py"
)
SPEC = util.spec_from_file_location("data_helpers", DATA_HELPERS_PATH)
assert SPEC is not None
data_helpers = util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(data_helpers)


class DataArgsA(BaseModel):
    train: str
    data_config_file: Path | None = None


class DataArgsB(BaseModel):
    val: str
    data_config_file: Path | None = None


class DataArgsWithAliases(BaseModel):
    train: str = Field(validation_alias=AliasChoices("train", "train_csv"))
    data_config_file: Path | None = None


def test_load_data_yaml_if_path__union_annotation(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "train": "train",
                "val": "val",
                "extra": "extra",
            }
        )
    )

    data = data_helpers.load_data_yaml_if_path(data_yaml, Union[DataArgsA, DataArgsB])

    assert data == {
        "train": "train",
        "val": "val",
        "data_config_file": data_yaml.resolve(),
    }


def test_load_data_yaml_if_path__single_annotation(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "train": "train",
                "val": "val",
                "extra": "extra",
            }
        )
    )

    data = data_helpers.load_data_yaml_if_path(data_yaml, DataArgsA)

    assert data == {"train": "train", "data_config_file": data_yaml.resolve()}


def test_load_data_yaml_if_path__keeps_validation_alias(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "train_csv": "train.csv",
                "extra": "extra",
            }
        )
    )

    data = data_helpers.load_data_yaml_if_path(data_yaml, DataArgsWithAliases)

    assert data == {"train_csv": "train.csv", "data_config_file": data_yaml.resolve()}


def test_load_data_yaml_if_path__direct_config_unchanged() -> None:
    data = {"train": "train"}

    assert data_helpers.load_data_yaml_if_path(data, DataArgsA) is data
