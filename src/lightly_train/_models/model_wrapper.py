#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    Protocol,
    overload,
    runtime_checkable,
)

import typing_extensions
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import NotRequired, Required, TypedDict, TypeVar

from lightly_train.types import PackageModel


class ForwardFeaturesOutput(TypedDict, total=False):
    """Output of the forward_features method."""

    features: Required[Tensor]
    cls_token: NotRequired[Tensor]


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
    def get_model(self) -> PackageModel:
        """Returns the model to store in the checkpoint."""
        ...


@runtime_checkable
class NNModule(Protocol):
    """Method definitions for nn.Module, directly copied from torch.nn.Module."""

    T_destination = TypeVar("T_destination", bound=Dict[str, Any])

    @overload
    def state_dict(
        self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...
    ) -> T_destination: ...

    @overload
    def state_dict(
        self, *, prefix: str = ..., keep_vars: bool = ...
    ) -> dict[str, Any]: ...

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None: ...


@runtime_checkable
class ModelWrapper(
    ForwardFeatures, ForwardPool, FeatureDim, ModelGetter, NNModule, Protocol
): ...


def missing_model_wrapper_attrs(
    model_wrapper: Any, exclude_module_attrs: bool = False
) -> list[str]:
    """Returns a list of attributes that are missing in the model wrapper.

    Args:
        model_wrapper:
            The model wrapper to check for missing attributes.
        exclude_module_attrs:
            If True, do not check attributes that are also in torch.nn.Module.
    """
    missing_attrs = []
    for attr in typing_extensions.get_protocol_members(ModelWrapper):
        if exclude_module_attrs and hasattr(Module, attr):
            continue
        if not hasattr(model_wrapper, attr):
            missing_attrs.append(attr)
    return sorted(missing_attrs)
