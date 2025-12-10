#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDETRObjectDetection,
)


def test_dinov3_ltdetr_load_train_state_dict_uses_ema_keys_and_forwards_kwargs() -> (
    None
):
    model = object.__new__(DINOv3LTDETRObjectDetection)

    captured: dict[str, Any] = {}

    def fake_load_state_dict(
        state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> str:
        captured["state_dict"] = state_dict
        captured["strict"] = strict
        captured["assign"] = assign
        return "successful"

    model.load_state_dict = fake_load_state_dict  # type: ignore[assignment]

    checkpoint_data = {
        "ema_model.model.backbone.weight": "some_weight_tensor",
        "model.backbone.weight": "ignored_tensor",
    }

    result = model.load_train_state_dict(checkpoint_data, strict=False, assign=True)

    assert result == "successful"

    assert captured["strict"] is False
    assert captured["assign"] is True

    assert captured["state_dict"] == {
        "backbone.weight": "some_weight_tensor",
        # Notice: "model.backbone.weight" is not here as it must be filtered out
    }
