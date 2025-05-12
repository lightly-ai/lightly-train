#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.nn import Module

from lightly_train._commands.common_helpers import is_global_rank_zero
from lightly_train._data.download import download_from_url
from lightly_train._modules.teachers.dinov2.configs import (
    load_and_merge_config,
)
from lightly_train._modules.teachers.dinov2.models import build_model_from_cfg

logger = logging.getLogger(__name__)


TEACHER_MODELS = {
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


def get_dinov2_teacher(teacher_name: str, checkpoint_dir: Path) -> Module:
    """Loads a DINOv2 teacher model and its pre-trained weights from a name.

    Returns the model in eval mode along with its embedding dimension.
    Raises a ValueError if the teacher name is unknown.
    """
    if teacher_name not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher: {teacher_name}")

    teacher_info = TEACHER_MODELS[teacher_name]
    url = teacher_info["url"]
    config_name = teacher_info["config"]

    # Load config.
    config_path = get_config_path(config_name)
    cfg = load_and_merge_config(str(config_path))

    # Build model.
    model, _, _ = build_model_from_cfg(cfg)
    model.eval()

    # Create the directory if it doesn't exist.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Cache the teacher checkpoint.
    checkpoint_path = checkpoint_dir / Path(url).name

    # Only the global rank zero downloads the checkpoint.
    if is_global_rank_zero():
        if not checkpoint_path.exists():
            logger.info(
                f"Downloading teacher weights from: '{url}' and saving them to: "
                f"'{checkpoint_path}'. The cache directory location can be configured "
                "with the LIGHTLY_TRAIN_CACHE_DIR environment variable."
            )
            download_from_url(url, checkpoint_path, timeout=180.0)

        else:
            logger.info(f"Using cached teacher weights from: '{checkpoint_path}'")

        # Load the checkpoint.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        logger.debug(f"Loaded teacher weights from '{checkpoint_path}'")

    return model


def get_config_path(config_name: str) -> Path:
    """Resolves a relative config path like 'eval/vitb14_pretrain
    into an absolute path relative to the configs package.
    """
    config_dir = Path(__file__).parent / "configs"
    full_path = config_dir / config_name
    return full_path
