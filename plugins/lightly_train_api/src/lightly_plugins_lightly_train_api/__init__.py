"""LightlyStudio plugins for the LightlyTrain API service."""

from __future__ import annotations

from lightly_plugins_lightly_train_api.inference_operator import (
    LightlyTrainApiInferenceOperator,
)
from lightly_plugins_lightly_train_api.training_operator import (
    LightlyTrainApiTrainingOperator,
)

__all__ = [
    "LightlyTrainApiInferenceOperator",
    "LightlyTrainApiTrainingOperator",
]
