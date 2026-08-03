#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train.types import PathLike


def resolve_data_paths(data_args: TaskDataArgs) -> None:
    base_dir = (
        data_args.data_config_file.parent
        if data_args.data_config_file is not None
        else Path.cwd()
    )
    data_args.resolve_data_paths(base_dir=base_dir)


def resolve_path(path: PathLike, base_dir: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
