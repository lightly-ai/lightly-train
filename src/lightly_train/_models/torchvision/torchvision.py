#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor
from torch.nn import Module

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    MultiScaleFeatureCNN,
)

# Spatial size of the dummy input used to read the multi-scale feature dimensions and
# strides. It is a multiple of the largest expected stride (32) so that the strides
# divide evenly.
_MULTI_SCALE_DUMMY_INPUT_SIZE = 224


class TorchvisionModelWrapper(Module, MultiScaleFeatureCNN):
    _torchvision_models: list[type[Module]]
    # Regex pattern for matching model names.
    _torchvision_model_name_pattern: str

    # Cache for the multi-scale feature dimensions and strides. Filled on first use.
    _multiscale_cache: tuple[list[int], list[int]] | None = None

    def multiscale_feature_dims(self) -> list[int]:
        dims, _ = self._get_multiscale_dims_and_strides()
        return dims

    def multiscale_feature_strides(self) -> list[int]:
        _, strides = self._get_multiscale_dims_and_strides()
        return strides

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        stages = self._extract_multiscale_stages(x)
        _validate_layer_indices(layer_indices=layer_indices, num_stages=len(stages))
        return [{"features": stages[index]} for index in layer_indices]

    def _extract_multiscale_stages(self, x: Tensor) -> list[Tensor]:
        """Returns the feature map of every multi-scale stage, in order.

        Model wrappers that support multi-scale features override this method. The
        default raises to signal that the architecture is not supported.
        """
        raise NotImplementedError(
            f"Multi-scale feature extraction is not supported for "
            f"'{type(self).__name__}'."
        )

    @classmethod
    def supports_multiscale_features(cls) -> bool:
        """True if the wrapper implements multi-scale feature extraction.

        A wrapper supports multi-scale features when it overrides
        ``_extract_multiscale_stages``; the base implementation raises.
        """
        return (
            cls._extract_multiscale_stages
            is not TorchvisionModelWrapper._extract_multiscale_stages
        )

    def _get_multiscale_dims_and_strides(self) -> tuple[list[int], list[int]]:
        """Returns the cached feature dimensions and strides, reading them on first use."""
        if self._multiscale_cache is None:
            self._multiscale_cache = _multiscale_dims_and_strides(
                model=self.get_model(),
                extract_stages=self._extract_multiscale_stages,
            )
        return self._multiscale_cache


def _multiscale_dims_and_strides(
    model: Module, extract_stages: Callable[[Tensor], list[Tensor]]
) -> tuple[list[int], list[int]]:
    """Reads the feature dimension and stride of every multi-scale stage.

    Runs a single dummy forward pass through ``extract_stages`` and reads the number of
    channels and the spatial stride of each stage from the output shapes. The model is
    set to eval mode for the forward pass so that batch norm statistics are not updated.
    The original mode is restored afterwards.

    Args:
        model:
            Model that provides the device for the dummy input.
        extract_stages:
            Function that returns the feature map of every stage for a given input.

    Returns:
        A tuple with the feature dimensions and the strides, one entry per stage.
    """
    device = next(model.parameters()).device
    size = _MULTI_SCALE_DUMMY_INPUT_SIZE
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            stages = extract_stages(torch.zeros(1, 3, size, size, device=device))
    finally:
        model.train(was_training)
    dims = [stage.shape[1] for stage in stages]
    strides = [size // stage.shape[-1] for stage in stages]
    return dims, strides


def _validate_layer_indices(layer_indices: Sequence[int], num_stages: int) -> None:
    """Makes sure that all layer indices are within the valid range.

    Args:
        layer_indices:
            Indices of the stages to extract features from.
        num_stages:
            Total number of multi-scale stages in the model.

    Raises:
        ValueError:
            If any index is not in the range ``[0, num_stages)``.
    """
    for index in layer_indices:
        if not 0 <= index < num_stages:
            raise ValueError(
                f"Layer index '{index}' is out of range for a model with "
                f"'{num_stages}' multi-scale stages."
            )
