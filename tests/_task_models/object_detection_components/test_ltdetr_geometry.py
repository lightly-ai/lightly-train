#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.object_detection_components.ltdetr_geometry import (
    ltdetr_image_size_divisor,
)


def test_ltdetr_image_size_divisor() -> None:
    assert ltdetr_image_size_divisor(14) == 28
    assert ltdetr_image_size_divisor(16) == 32
