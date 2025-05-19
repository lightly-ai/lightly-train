#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch.nn import Module

from lightly_train._data.cache import get_cache_dir


def _parse_model_name(model: str) -> tuple[str, str]:
    parts = model.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Model name has incorrect format. Should be 'package/model' but is "
            f"'{model}'"
        )
    package_name = parts[0]
    model_name = parts[1]
    return package_name, model_name


def get_teacher(teacher_name: str) -> Module:
    """Loads a teacher model and its pre-trained weights from a name.

    Returns the model in eval mode along with its embedding dimension.
    Raises a ValueError if the teacher name is unknown.
    """
    # Get the cache directory.
    cache_dir = get_cache_dir()

    # Infer the sub-directory for the teacher weights
    checkpoint_dir = cache_dir / "weights"

    # Get the teacher model.
    package_name, model_name = _parse_model_name(teacher_name)
    if package_name == "dinov2_vit":
        from lightly_train._modules.teachers.dinov2.build_teacher import (
            get_dinov2_teacher,
        )

        return get_dinov2_teacher(model_name, checkpoint_dir)
    else:
        raise ValueError(f"Unknown teacher: '{teacher_name}'")
