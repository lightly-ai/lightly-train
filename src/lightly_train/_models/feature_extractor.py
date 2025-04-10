#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor
from torch.nn import Module
from typing_extensions import Required, TypedDict


class ForwardFeaturesOutput(TypedDict, total=False):
    """Output of the forward_features method."""

    features: Required[Tensor]


class ForwardPoolOutput(TypedDict, total=False):
    """Output of the forward_pool method."""

    pooled_features: Required[Tensor]


@runtime_checkable
class ForwardFeatures(Protocol):
    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        """Extracts features.

        Args:
            x: Inputs with shape (B, C_in, H_in, W_in).

        Returns:
            Dict with "features" entry containing the extracted features. The features
            have shape (B, feature_dim, H_out, W_out). H_out and W_out are usually >1.
        """
        ...


@runtime_checkable
class ForwardPool(Protocol):
    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        """Pools features, should be called after `forward_features`.

        Args:
            x:
                Output of `forward_features` method. Must be a dict with a "features"
                entry containing the extracted features with shape
                (B, feature_dim, H_in, W_in).

        Returns:
            Dict with "pooled_features" entry containing the pooled features with shape
            (B, feature_dim, H_out, W_out). H_out and W_out depend on the pooling
            strategy but are usually 1.
        """
        ...


@runtime_checkable
class FeatureDim(Protocol):
    def feature_dim(self) -> int:
        """Returns the feature dimension of the extractor."""
        ...


@runtime_checkable
class ModelGetter(Protocol):
    def get_model(self) -> Module:
        """Returns the model to store in the checkpoint."""
        ...


@runtime_checkable
class FeatureExtractor(ForwardFeatures, ForwardPool, FeatureDim, Protocol): ...
