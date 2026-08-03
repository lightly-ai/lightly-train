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
    Literal,
    Mapping,
    Protocol,
    Sequence,
    overload,
    runtime_checkable,
)

import typing_extensions
from torch import Tensor
from torch.nn import Module, Parameter
from typing_extensions import NotRequired, Required, Self, TypedDict, TypeVar

from lightly_train.types import PackageModel


class ForwardFeaturesOutput(TypedDict, total=False):
    """Output of the forward_features method."""

    features: Required[Tensor]
    cls_token: NotRequired[Tensor]


class ForwardPoolOutput(TypedDict, total=False):
    """Output of the forward_pool method."""

    pooled_features: Required[Tensor]


class ArchitectureInfo(TypedDict):
    """Architecture information for the model wrapper."""

    model_type: Literal["convolutional", "transformer", "hybrid"]
    norm_type: Literal["batchnorm", "layernorm"]


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
class ArchitectureInfoGettable(Protocol):
    def architecture_info(self) -> ArchitectureInfo:
        """Returns architecture information for the model wrapper."""
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

    def eval(self) -> Self: ...

    def modules(self) -> Iterator[Module]: ...

    def requires_grad_(self, requires_grad: bool = True) -> Self: ...


@runtime_checkable
class ModelWrapper(
    ForwardFeatures,
    ForwardPool,
    FeatureDim,
    ModelGetter,
    NNModule,
    Protocol,
): ...


@runtime_checkable
class MultiScaleFeatureDims(Protocol):
    def multiscale_feature_dims(self) -> list[int]:
        """Returns the feature dimensions of each layer/stage in the model.

        The returned list has one entry per layer/stage, indexed from 0 (earliest
        layer/stage) to N-1 (last layer/stage). For a ViT all entries are typically
        the same (equal to ``embed_dim``). For a ConvNeXt each stage has a different
        dimension (e.g. [96, 192, 384, 768]).

        The index of each entry corresponds to the layer indices accepted by
        ``forward_multiscale_features``.
        """
        ...


@runtime_checkable
class PatchSize(Protocol):
    def patch_size(self) -> int:
        """Returns the patch size of the model.

        For ViT models this is the size of each patch (e.g., 16 or 14).
        """
        ...


@runtime_checkable
class MultiScaleFeatureStrides(Protocol):
    def multiscale_feature_strides(self) -> list[int]:
        """Returns the feature strides of each layer/stage in the model.

        The returned list has one entry per layer/stage, indexed from 0 (earliest
        layer/stage) to N-1 (last layer/stage). For a ViT all entries are typically
        the same (equal to the patch size). For a ConvNeXt each stage has a different
        stride (e.g. [4, 8, 16, 32] for a model with patch size 4).

        The index of each entry corresponds to the layer indices accepted by
        ``forward_multiscale_features``.
        """
        ...


@runtime_checkable
class ForwardMultiScaleFeatures(Protocol):
    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        """Extracts multi-scale features from the specified layers/stages.

        The ``layer_indices`` are 0-based indices from the beginning of the network,
        corresponding to the indices of ``multiscale_feature_dims()``. For a ViT each
        index refers to a transformer block. For a ConvNeXt each index refers to a
        stage (downsample stage + residual blocks).

        Args:
            x: Inputs with shape (B, C_in, H_in, W_in).
            layer_indices: Indices of the layers/stages to extract features from.

        Returns:
            List of dicts, one per requested layer/stage, in the same order as
            ``layer_indices``. Each dict has a ``"features"`` entry containing the
            extracted features with shape (B, feature_dim, H_out, W_out), and
            optionally a ``"cls_token"`` entry. Features are normalized. Different
            entries may have different feature dimensions and spatial resolutions.
        """
        ...


class MultiScaleFeatureViT(
    ModelWrapper, ForwardMultiScaleFeatures, MultiScaleFeatureDims, PatchSize, Protocol
):
    """Protocol for ViT models with multiscale feature extraction."""


class MultiScaleFeatureCNN(
    ModelWrapper,
    ForwardMultiScaleFeatures,
    MultiScaleFeatureDims,
    MultiScaleFeatureStrides,
    Protocol,
):
    """Protocol for CNN models with multiscale feature extraction."""


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
