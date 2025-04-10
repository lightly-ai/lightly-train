#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf, SCMode


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    config_dict = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
        enum_to_str=False,
        structured_config_mode=SCMode.DICT,
    )
    assert isinstance(config_dict, dict)
    # Type ignore required because OmegaConf.to_container() contains more possible
    # types than just dict[str, Any] but we know that it will always return string keys.
    result: dict[str, Any] = config_dict  # type: ignore[assignment]
    return result
