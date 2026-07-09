#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Union, get_args, get_origin

import fsspec
import yaml


def _resolve_path_relative_to_yaml(value: Any, yaml_path: Path) -> Any:
    """Resolve a relative top-level data path against the YAML file location."""
    if not isinstance(value, dict) or "path" not in value:
        return value
    path = value["path"]
    if not isinstance(path, (str, Path)):
        return value
    path = Path(path)
    if path.is_absolute():
        return value
    return {**value, "path": yaml_path.parent / path}


def load_data_yaml_if_path(value: Any, data_annotation: Any) -> Any:
    """Loads a data config from a YAML file if ``value`` is a path.

    If ``value`` is a string or ``Path`` it is interpreted as the path to a YAML file
    that is loaded and returned as a dictionary. For local YAML files, a relative
    top-level path value is resolved against the YAML file's parent directory.
    All keys that are not part of the Pydantic model are ignored. As the data
    config can be a ``Union``, it would be
    impossible to figure out which keys to exclude, so in that case the fields of all
    union members are included. If ``value`` is not a path it is returned unchanged.

    Args:
        value:
            The value of the ``data`` field. Either a path to a YAML file or an
            already-parsed config (dict or model instance).
        data_annotation:
            The type annotation of the ``data`` field. Usually obtained via
            ``cls.model_fields["data"].annotation``.
    """
    if isinstance(value, (str, Path)):
        yaml_path_or_url = str(value)
        with fsspec.open(value, "r") as file:
            value = yaml.safe_load(file)
        if fsspec.utils.infer_storage_options(yaml_path_or_url)["protocol"] == "file":
            value = _resolve_path_relative_to_yaml(value, Path(yaml_path_or_url))
        if get_origin(data_annotation) is Union:
            members = get_args(data_annotation)
        else:
            members = (data_annotation,)
        data_attributes = {
            name
            for m in members
            for name in m.model_fields  # type: ignore[attr-defined]
        }
        value = {name: val for name, val in value.items() if name in data_attributes}
    return value


def set_default_data_format(value: Any, default: str = "yolo") -> Any:
    """Sets a default ``format`` on a data config dict if none is given.

    Returns ``value`` unchanged if it is not a dict or already has a ``format`` key.
    """
    if isinstance(value, dict) and "format" not in value:
        value = {**value, "format": default}
    return value
