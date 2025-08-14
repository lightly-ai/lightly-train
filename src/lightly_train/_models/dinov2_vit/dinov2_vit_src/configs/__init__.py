#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# Modifications Copyright 2025 Lightly AG:
# - added get_config_path function
# - added MODELS dictionary

from __future__ import annotations

import pathlib

from omegaconf import OmegaConf
from typing_extensions import NotRequired, TypedDict


class _Model(TypedDict):
    """Dictionary type for DINOv2 models."""

    url: str
    config: str
    list: bool
    alias_for: NotRequired[str]


MODELS: dict[str, _Model] = {
    # Test model for development purposes only.
    "_vittest14": _Model(
        url="",
        config="train/_vittest14",
        list=False,
    ),
    # Default models are pretrained and with registers.
    "vits14": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
        config="eval/vits14_reg4_pretrain",
        list=True,
    ),
    "vitb14": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
        config="eval/vitb14_reg4_pretrain",
        list=True,
    ),
    "vitl14": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
        config="eval/vitl14_reg4_pretrain",
        list=True,
    ),
    "vitg14": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
        config="eval/vitg14_reg4_pretrain",
        list=True,
    ),
    # Models without registers if needed.
    "vits14-noreg": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        config="eval/vits14_pretrain",
        list=False,
    ),
    "vitb14-noreg": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        config="eval/vitb14_pretrain",
        list=False,
    ),
    "vitl14-noreg": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        config="eval/vitl14_pretrain",
        list=False,
    ),
    "vitg14-noreg": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        config="eval/vitg14_pretrain",
        list=False,
    ),
    # Not pretrained models if needed.
    "vits14-notpretrained": _Model(
        url="",
        config="train/vits14_reg4",
        list=False,
    ),
    "vitb14-notpretrained": _Model(
        url="",
        config="train/vitb14_reg4",
        list=False,
    ),
    "vitl14-notpretrained": _Model(
        url="",
        config="train/vitl14_reg4",
        list=False,
    ),
    "vitg14-notpretrained": _Model(
        url="",
        config="train/vitg14_reg4",
        list=False,
    ),
    "vits14-noreg-notpretrained": _Model(
        url="",
        config="train/vits14",
        list=False,
    ),
    "vitb14-noreg-notpretrained": _Model(
        url="",
        config="train/vitb14",
        list=False,
    ),
    "vitl14-noreg-notpretrained": _Model(
        url="",
        config="train/vitl14",
        list=False,
    ),
    "vitg14-noreg-notpretrained": _Model(
        url="",
        config="train/vitg14",
        list=False,
    ),
    # Models with `-pretrained` suffix for backwards compatibility.
    "vits14-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
        config="eval/vits14_reg4_pretrain",
        list=False,
        alias_for="vits14",
    ),
    "vitb14-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
        config="eval/vitb14_reg4_pretrain",
        list=False,
        alias_for="vitb14",
    ),
    "vitl14-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
        config="eval/vitl14_reg4_pretrain",
        list=False,
        alias_for="vitl14",
    ),
    "vitg14-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
        config="eval/vitg14_reg4_pretrain",
        list=False,
        alias_for="vitg14",
    ),
    "vits14-noreg-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        config="eval/vits14_pretrain",
        list=False,
        alias_for="vits14-noreg",
    ),
    "vitb14-noreg-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        config="eval/vitb14_pretrain",
        list=False,
        alias_for="vitb14-noreg",
    ),
    "vitl14-noreg-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        config="eval/vitl14_pretrain",
        list=False,
        alias_for="vitl14-noreg",
    ),
    "vitg14-noreg-pretrained": _Model(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        config="eval/vitg14_pretrain",
        list=False,
        alias_for="vitg14-noreg",
    ),
}


def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)


def load_and_merge_config(config_name: str):
    dinov2_default_config = load_config("ssl_default_config")
    default_config = OmegaConf.create(dinov2_default_config)
    loaded_config = load_config(config_name)
    return OmegaConf.merge(default_config, loaded_config)


def get_config_path(config_name: str) -> pathlib.Path:
    """Resolves a relative config path like 'eval/vitb14_pretrain
    into an absolute path relative to the configs package.
    """
    config_dir = pathlib.Path(__file__).parent
    full_path = config_dir / config_name
    return full_path
