#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.export import Dim
from torch.nn import Module

from lightly_train._pre_post_processors.base import (
    BaseModelOutput,
    DynamoExportMixin,
    ModelInputSpec,
    TensorSpec,
)


class SemSegPreProcessor(Module):
    def forward(self, image: Tensor) -> tuple[Tensor]:
        return (image,)


@dataclass
class SemanticSegmentationSoftMaxPostProcessorOutput(BaseModelOutput):
    probabilities: Tensor


class SemanticSegmentationSoftMaxPostProcessor(Module, DynamoExportMixin):
    """Post processor for semantic segmentation that applies a softmax to the output of the model.

    This is useful for models that output logits, and we want to convert them to probabilities.
    """

    @property
    def model_input_spec(self) -> ModelInputSpec:
        return ModelInputSpec(
            input_specs={
                "logits": TensorSpec(
                    shape=(3, 224, 224), dtype=torch.float32, is_batched=True
                )
            },
            input_dynamic_shapes={
                "logits": (Dim.DYNAMIC, Dim.STATIC, Dim.STATIC, Dim.STATIC)
            },
        )

    def forward(  # type: ignore[override]
        self, logits: Tensor
    ) -> SemanticSegmentationSoftMaxPostProcessorOutput:
        """Applies a softmax to the input tensor.

        Args:
            logits: The input tensor of shape (batch_size, num_classes, height, width).

        Returns:
            The output tensor of shape (batch_size, num_classes, height, width) with softmax applied.
        """
        return SemanticSegmentationSoftMaxPostProcessorOutput(
            probabilities=torch.softmax(logits, dim=1)
        )
