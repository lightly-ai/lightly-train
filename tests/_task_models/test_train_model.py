#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, cast

from lightly_train._task_models.dinov3_ltdetr_object_detection.train_model import (
    DINOv3LTDETRObjectDetectionTrain,
)


class FakeTaskModel:
    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> str:
        self.captured["state_dict"] = state_dict
        self.captured["strict"] = strict
        self.captured["assign"] = assign
        return "incompatible_keys"

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        """Also capture load_state_dict calls."""
        self.captured["state_dict"] = state_dict
        self.captured["strict"] = strict
        self.captured["assign"] = assign

    def state_dict(self) -> dict[str, Any]:
        """Simulates the main model's current weights."""
        return {
            "backbone.weight": "main_model_backbone_tensor",
            "head.bias": "main_model_head_tensor",
        }


class FakeEmaModel:
    """A fake EMA model wrapper."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}
        self.model = FakeTaskModel()

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
        self.captured["state_dict"] = state_dict
        self.captured["strict"] = strict
        self.captured["assign"] = assign


def test_ltdetr_load_train_state_dict_scenarios() -> None:
    """Verifies that the TrainModel correctly coordinates loading between the main model and EMA model."""

    train_model = object.__new__(DINOv3LTDETRObjectDetectionTrain)

    # -------------------------------------------------------------------------
    # Scenario 1: EMA model is None
    # -------------------------------------------------------------------------
    train_model.model = FakeTaskModel()  # type: ignore[assignment]
    train_model.ema_model = None

    checkpoint = {"some_key": "some_tensor"}

    train_model.load_train_state_dict(checkpoint, strict=False)

    # The main model should receive exactly the provided state_dict
    assert train_model.model.captured["state_dict"] == checkpoint  # type: ignore[index]
    assert train_model.model.captured["strict"] is False  # type: ignore[index]

    # -------------------------------------------------------------------------
    # Scenario 2: EMA model exists and checkpoint has EMA weights
    # -------------------------------------------------------------------------
    train_model.model = FakeTaskModel()  # type: ignore[assignment]
    train_model.ema_model = FakeEmaModel()  # type: ignore[assignment]

    checkpoint = {"model.key": "main_model_tensor", "ema_model.key": "ema_tensor"}

    train_model.load_train_state_dict(checkpoint, strict=False)

    # Main model should get the full checkpoint (it handles filtering internally)
    assert train_model.model.captured["state_dict"] == checkpoint  # type: ignore[index]

    # EMA model should get only the stripped "ema_model" keys
    expected_ema_dict = {"key": "ema_tensor"}
    ema_model = cast(FakeEmaModel, train_model.ema_model)
    assert ema_model.captured["state_dict"] == expected_ema_dict
    assert ema_model.captured["strict"] is False

    # -------------------------------------------------------------------------
    # Scenario 3: EMA model exists but checkpoint has NO EMA weights
    # -------------------------------------------------------------------------
    train_model.model = FakeTaskModel()  # type: ignore[assignment]
    train_model.ema_model = FakeEmaModel()  # type: ignore[assignment]

    # Checkpoint has NO keys starting with "ema_model."
    checkpoint = {"model.key": "main_model_tensor"}

    train_model.load_train_state_dict(checkpoint, strict=False)

    # Main model gets loaded normally
    assert train_model.model.captured["state_dict"] == checkpoint  # type: ignore[index]

    # EMA model should NOT have loaded from checkpoint
    ema_model = cast(FakeEmaModel, train_model.ema_model)
    assert "state_dict" not in ema_model.captured

    # Instead, the EMA model should have loaded the main model's state_dict
    expected_synced_weights = {
        "backbone.weight": "main_model_backbone_tensor",
        "head.bias": "main_model_head_tensor",
    }
    assert ema_model.model.captured["state_dict"] == expected_synced_weights
