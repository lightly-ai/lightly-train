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
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.export.dynamic_shapes import Dim, _DimHint

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_specs: dict[str, TensorSpec] = Field(
        ..., description="Mapping of input names to their corresponding TensorSpec."
    )
    input_dynamic_shapes: dict[str, tuple[_DimHint | Dim, ...]] = Field(
        ...,
        description=(
            "Mapping of input names to tuples of Dim enums, indicating which dimensions "
            "are dynamic or static."
        ),
    )

    @model_validator(mode="after")
    def _validate_input_dynamic_shapes(self) -> "ModelInputSpec":
        for name, spec in self.input_specs.items():
            dynamic_shapes = self.input_dynamic_shapes[name]
            expected_rank = len(spec.shape) + int(spec.is_batched)
            if len(dynamic_shapes) != expected_rank:
                raise ValueError(
                    f"input_dynamic_shapes for '{name}' has rank "
                    f"{len(dynamic_shapes)}, expected {expected_rank}."
                )

            for dim_index, dim in enumerate(dynamic_shapes):
                if spec.is_batched and dim_index == 0:
                    continue
                if not self._is_static_dim(dim):
                    raise ValueError(
                        f"input_dynamic_shapes for '{name}' contains a dynamic "
                        f"dimension at index {dim_index}. Only the batch dimension "
                        "may be dynamic."
                    )

        return self

    def example_inputs(self, batch_size: int | None = None) -> dict[str, Tensor]:
        """Generate example inputs based on the specified input specs.

        If an input is marked as batched, the example tensor uses the given batch
        size or the minimum dynamic batch size if available.

        Returns:
            A dictionary mapping input names to example tensors.
        """
        inputs = {}
        for name, spec in self.input_specs.items():
            if spec.is_batched:
                inputs[name] = spec.example_tensor(
                    batch_size=(
                        batch_size
                        if batch_size is not None
                        else self._default_batch_size(name)
                    )
                )
            else:
                inputs[name] = spec.example_tensor()
        return inputs

    @staticmethod
    def _is_static_dim(dim: _DimHint | Dim) -> bool:
        return isinstance(dim, _DimHint) and dim.type.name == "STATIC"

    def _default_batch_size(self, name: str) -> int:
        batch_dim = self.input_dynamic_shapes[name][0]
        minimum = getattr(batch_dim, "min", None)
        return 1 if minimum is None else minimum


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
