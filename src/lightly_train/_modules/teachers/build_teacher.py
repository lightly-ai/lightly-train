#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.nn import Module

from lightly_train._data.cache import get_cache_dir


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
    if teacher_name.startswith("dinov2_"):
        from lightly_train._modules.teachers.dinov2.build_teacher import (
            get_dinov2_teacher,
        )

        return get_dinov2_teacher(teacher_name, checkpoint_dir)
    else:
        raise ValueError(f"Unknown teacher: '{teacher_name}'")
