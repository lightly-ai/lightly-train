#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.nn import Module

from lightly_train._modules.teachers.dinov2.configs import (
    load_and_merge_config,
)
from lightly_train._modules.teachers.dinov2.models import build_model_from_cfg

logger = logging.getLogger(__name__)


TEACHER_MODELS = {
    "dinov2_vitb14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "config": "eval/vitb14_pretrain",
    },
}


def get_dinov2_teacher(teacher_name: str) -> Module:
    """Loads a DINOv2 teacher model and its pre-trained weights from a name.

    Returns the model in eval mode along with its embedding dimension.
    Raises a ValueError if the teacher name is unknown.
    """
    if teacher_name not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher: {teacher_name}")

    teacher_info = TEACHER_MODELS[teacher_name]
    url = teacher_info["url"]
    config_name = teacher_info["config"]

    # Load config
    config_path = get_config_path(config_name)
    cfg = load_and_merge_config(config_path)

    # Build model
    model, _, _ = build_model_from_cfg(cfg)
    model.eval()

    # Cache checkpoint
    cache_dir = Path.home() / ".cache" / "lightlytrain" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / Path(url).name

    if not checkpoint_path.exists():
        logger.info(f"Downloading teacher weights from: '{url}'")
        urlretrieve(url, checkpoint_path)
    else:
        logger.info(f"Using cached teacher weights from: '{checkpoint_path}'")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    logger.info(f"Loaded teacher weights from '{checkpoint_path}'")

    return model


def get_config_path(config_name: str) -> str:
    """Resolves a relative config path like 'eval/vitb14_pretrain
    into an absolute path relative to the configs package.
    """
    config_dir = Path(__file__).parent / "configs"
    full_path = config_dir / config_name
    return str(full_path)
