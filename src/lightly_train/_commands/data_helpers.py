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
from typing_extensions import Annotated


def _iter_model_fields(data_annotation: Any) -> set[str]:
    if get_origin(data_annotation) is Annotated:
        data_annotation = get_args(data_annotation)[0]
    if get_origin(data_annotation) is Union:
        members = get_args(data_annotation)
    else:
        members = (data_annotation,)
    return {
        name
        for member in members
        for name in member.model_fields  # type: ignore[attr-defined]
    }


def _filter_data_attributes(value: Any, data_annotation: Any) -> Any:
    if not isinstance(value, dict):
        return value
    data_attributes = _iter_model_fields(data_annotation)
    return {name: val for name, val in value.items() if name in data_attributes}


def _resolve_relative_path(path: Any, base_dir: Path) -> Any:
    if not isinstance(path, (str, Path)):
        return path
    path = Path(path)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_object_detection_paths_relative_to_yaml(
    value: Any, yaml_path: Path
) -> Any:
    if not isinstance(value, dict):
        return value

    data_format = value.get("format", "yolo")
    base_dir = yaml_path.parent
    if data_format == "yolo":
        if "path" not in value:
            return value
        return {**value, "path": _resolve_relative_path(value["path"], base_dir)}
    if data_format == "coco":
        value = {**value}
        for split in ("train", "val", "test"):
            split_value = value.get(split)
            if isinstance(split_value, dict) and "annotations" in split_value:
                value[split] = {
                    **split_value,
                    "annotations": _resolve_relative_path(
                        split_value["annotations"], base_dir
                    ),
                }
        return value
    return value


def prepare_object_detection_data(value: Any, data_annotation: Any) -> Any:
    """Prepare object detection data configs.

    Supports either an already parsed data config or a path to a YAML file. Local
    YAML file paths are used as the base for relative YOLO path values and
    relative COCO annotation file paths. Unknown YAML keys are ignored.
    """
    if isinstance(value, (str, Path)):
        yaml_path_or_url = str(value)
        with fsspec.open(value, "r") as file:
            value = yaml.safe_load(file)
        if fsspec.utils.infer_storage_options(yaml_path_or_url)["protocol"] == "file":
            value = _resolve_object_detection_paths_relative_to_yaml(
                value, Path(yaml_path_or_url)
            )
        value = _filter_data_attributes(value, data_annotation)
    return set_default_data_format(value)


def set_default_data_format(value: Any, default: str = "yolo") -> Any:
    """Sets a default format on a data config dict if none is given.

    Returns value unchanged if it is not a dict or already has a format key.
    """
    if isinstance(value, dict) and "format" not in value:
        value = {**value, "format": default}
    return value
