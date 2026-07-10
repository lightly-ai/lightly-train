#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
import logging
from abc import ABC
from dataclasses import dataclass, fields
from typing import (
    Any,
)

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.export.dynamic_shapes import _DimHint

logger = logging.getLogger(__name__)


class TensorSpec(BaseModel):
    """Specification of a single tensor that can be used for model inputs or outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shape: tuple[int, ...] = Field(
        ..., description="Shape of the tensor excluding batch dimension."
    )
    dtype: torch.dtype = Field(..., description="Data type of the tensor.")
    is_batched: bool = Field(
        ..., description="Indicates whether the tensor appears batched in the model."
    )

    def example_tensor(self, batch_size: int | None = None) -> Tensor:
        """Generate an example tensor based on the specified shape and dtype.

        Returns:
            A tensor with the defined shape and data type.
        """
        if batch_size is not None and self.is_batched:
            shape = (batch_size, *self.shape)
        else:
            shape = self.shape
        return torch.zeros(shape, dtype=self.dtype)


class ModelInputSpec(BaseModel):
    """Specification of the model's inputs."""

    input_specs: dict[str, TensorSpec] = Field(
        ..., description="Mapping of input names to their corresponding TensorSpec."
    )
    input_dynamic_shapes: dict[str, tuple[_DimHint, ...]] = Field(
        ...,
        description=(
            "Mapping of input names to tuples of Dim enums, indicating which dimensions "
            "are dynamic or static."
        ),
    )

    def example_inputs(self) -> dict[str, Tensor]:
        """Generate example inputs based on the specified input specs.

        If an input is marked as batched, the example tensor will be unsqueezed at the
        0-th dimension to simulate a batch of size 1.

        Returns:
            A dictionary mapping input names to example tensors.
        """
        inputs = {}
        for name, spec in self.input_specs.items():
            if spec.is_batched:
                inputs[name] = spec.example_tensor().unsqueeze(0)
            else:
                inputs[name] = spec.example_tensor()
        return inputs


@dataclass
class BaseModelOutput(ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        torch.utils._pytree.register_pytree_node(
            cls,
            _model_output_flatten,
            functools.partial(_model_output_unflatten, output_type=cls),
        )


def _model_output_flatten(
    model_output: BaseModelOutput,
) -> tuple[list[Tensor], list[str]]:
    values = []
    context = []
    for field in fields(model_output):
        values.append(getattr(model_output, field.name))
        context.append(field.name)
    return values, context


def _model_output_unflatten(
    values: list[Tensor],
    context: list[str],
    output_type: type[BaseModelOutput],
) -> BaseModelOutput:
    field_values = {name: value for name, value in zip(context, values)}
    return output_type(**field_values)


def output_names_from_model_output(output: BaseModelOutput) -> list[str]:
    """Derive ONNX output names from a BaseModelOutput.

    The dataclass field names of a ``BaseModelOutput`` (in declaration order) are used
    as the ONNX output names. This is the same source of truth used by the pytree
    flattening (`_model_output_flatten`), so the output order is guaranteed to match.

    Args:
        output: A ``BaseModelOutput`` instance returned by the model's forward.

    Returns:
        The field names of the output dataclass, in declaration order.
    """
    return [field.name for field in fields(output)]
