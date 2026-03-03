#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.semantic_segmentation_multihead.task_model import (
    SemanticSegmentationMultihead,
)
from lightly_train._task_models.semantic_segmentation_multihead.train_model import (
    SemanticSegmentationMultiheadTrain,
    SemanticSegmentationMultiheadTrainArgs,
)

__all__ = [
    "SemanticSegmentationMultihead",
    "SemanticSegmentationMultiheadTrain",
    "SemanticSegmentationMultiheadTrainArgs",
]
