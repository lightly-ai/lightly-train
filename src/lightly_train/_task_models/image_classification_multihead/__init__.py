#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.image_classification_multihead.task_model import (
    ImageClassificationMultihead,
)
from lightly_train._task_models.image_classification_multihead.train_model import (
    ImageClassificationMultiheadTrain,
    ImageClassificationMultiheadTrainArgs,
)

__all__ = [
    "ImageClassificationMultihead",
    "ImageClassificationMultiheadTrain",
    "ImageClassificationMultiheadTrainArgs",
]
