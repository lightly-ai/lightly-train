#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from lightly_train._modules.teachers.dinov2.dinov2_helper import load_weights
from torch.nn import Module

from lightly_train._modules.teachers.dinov2.configs import MODELS as TEACHER_MODELS
from lightly_train._modules.teachers.dinov2.configs import (
    get_config_path,
    load_and_merge_config,
)
from lightly_train._modules.teachers.dinov2.models import build_model_from_cfg




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

    model = load_weights(model=model, checkpoint_dir=checkpoint_dir, url=url)

    return model
