#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

from lightly_train._configs import validate
from lightly_train._configs.config import PydanticConfig
from lightly_train.types import PathLike


def train_task(
    *,
    out: PathLike,
    data: dict[str, Any],
    model: str,
    task: Literal["semantic_segmentation"],
) -> None:
    config = validate.pydantic_model_validate(TrainTaskConfig, locals())
    train_task_from_config(config=config)


def train_task_from_config(config: TrainTaskConfig) -> None:
    # Training loop goes here.
    # Put functions into train_task_helpers.py
    pass


class TrainTaskConfig(PydanticConfig):
    out: PathLike
    data: dict[str, Any]
    model: str
    task: Literal["semantic_segmentation"]
