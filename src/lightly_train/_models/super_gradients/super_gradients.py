#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from abc import ABC, abstractmethod

from torch.nn import Module

from lightly_train._models.feature_extractor import FeatureExtractor


class SuperGradientsFeatureExtractor(ABC, FeatureExtractor):
    @classmethod
    @abstractmethod
    def is_supported_model_cls(cls, model_cls: type[Module]) -> bool: ...

    @classmethod
    @abstractmethod
    def supported_model_classes(cls) -> tuple[type[Module], ...]: ...
