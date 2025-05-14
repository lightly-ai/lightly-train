#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# Modifications Copyright 2025 Lightly AG:
# - added get_config_path function
# - added MODELS and TRAIN_MODELS dictionaries

import pathlib

from omegaconf import OmegaConf

MODELS = {
    "dinov2_vits14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "config": "eval/vits14_pretrain",
    },
    "dinov2_vitb14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "config": "eval/vitb14_pretrain",
    },
    "dinov2_vitl14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        "config": "eval/vitl14_pretrain",
    },
    "dinov2_vitg14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        "config": "eval/vitg14_pretrain",
    },
}

TRAIN_MODELS = {
    "vits14": {
        "config": "train/vits14",
    },
    "vitb14": {
        "config": "train/vitb14",
    },
    "vitl14": {
        "config": "train/vitl14",
    },
    "vitg14": {
        "config": "train/vitg14",
    },
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
