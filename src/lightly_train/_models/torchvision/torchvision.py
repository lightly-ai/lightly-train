#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Sequence

from torch.nn import Conv2d, Module

from lightly_train._models.model_wrapper import ModelWrapper


class TorchvisionModelWrapper(Module, ModelWrapper):
    _torchvision_models: list[type[Module]]
    # Regex pattern for matching model names.
    _torchvision_model_name_pattern: str


def _validate_layer_indices(layer_indices: Sequence[int], num_layers: int) -> None:
    """Makes sure that all layer indices are within the valid range.

    Negative indices are rejected because the wrappers index a list of stage outputs,
    where a negative index would silently select a stage instead of raising.

    Args:
        layer_indices:
            Indices of the stages to extract features from.
        num_layers:
            Total number of multi-scale stages in the model.

    Raises:
        ValueError:
            If any index is not in the range ``[0, num_layers)``.
    """
    for layer_index in layer_indices:
        if not 0 <= layer_index < num_layers:
            raise ValueError(
                f"Layer index {layer_index} is out of range, it must be in "
                f"[0, {num_layers - 1}]."
            )


def _last_conv_out_channels(module: Module) -> int:
    """Returns the number of output channels of the last Conv2d in the module."""
    convs = [m for m in module.modules() if isinstance(m, Conv2d)]
    return convs[-1].out_channels


def _max_conv_stride(module: Module) -> int:
    """Returns the largest stride of any Conv2d in the module.

    Blocks that downsample apply the same stride on every branch, so the maximum is the
    stride of the block as a whole. Blocks that do not downsample return 1.
    """
    return max(m.stride[0] for m in module.modules() if isinstance(m, Conv2d))
