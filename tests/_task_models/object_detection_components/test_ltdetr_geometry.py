from __future__ import annotations

from lightly_train._task_models.object_detection_components.ltdetr_geometry import (
    ltdetr_image_size_divisor,
)


def test_ltdetr_image_size_divisor() -> None:
    assert ltdetr_image_size_divisor(14) == 28
    assert ltdetr_image_size_divisor(16) == 32
