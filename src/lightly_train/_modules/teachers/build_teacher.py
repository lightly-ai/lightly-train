#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Tuple

from torch.nn import Module


def get_teacher(teacher_name: str) -> tuple[Module, int]:
    """Loads a teacher model and its pre-trained weights from a name.

    Returns the model in eval mode along with its embedding dimension.
    Raises a ValueError if the teacher name is unknown.
    """
    if teacher_name.startswith("dinov2_"):
        from lightly_train._modules.teachers.dinov2 import get_dinov2_teacher

        return get_dinov2_teacher(teacher_name)
    else:
        raise ValueError(f"Unknown teacher: {teacher_name}")
