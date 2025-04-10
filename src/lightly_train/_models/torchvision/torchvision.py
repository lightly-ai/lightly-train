#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch.nn import Module

from lightly_train._models.feature_extractor import FeatureExtractor


class TorchvisionFeatureExtractor(Module, FeatureExtractor):
    _torchvision_models: list[type[Module]]
    # Regex pattern for matching model names.
    _torchvision_model_name_pattern: str
