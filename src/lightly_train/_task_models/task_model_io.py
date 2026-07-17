#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import functools
from abc import ABC
from dataclasses import dataclass, fields
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.export.dynamic_shapes import Dim, _DimHint
from typing_extensions import Self


class TensorSpec(BaseModel):
    """Specification of a single tensor used for model inputs or outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shape: tuple[int, ...] = Field(
        ..., description="Shape of the tensor excluding its batch dimension."
    )
    dtype: torch.dtype = Field(..., description="Data type of the tensor.")
    is_batched: bool = Field(
        ..., description="Whether the tensor appears batched in the model."
    )

    def example_tensor(self, batch_size: int | None = None) -> Tensor:
        shape = (
            (batch_size, *self.shape)
            if batch_size is not None and self.is_batched
            else self.shape
        )
        return torch.zeros(shape, dtype=self.dtype)


class ModelInputSpec(BaseModel):
    """Specification of a model's named tensor inputs and dynamic dimensions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_specs: dict[str, TensorSpec]
    input_dynamic_shapes: dict[str, tuple[_DimHint | Dim, ...]]

    @model_validator(mode="after")
    def _validate_input_dynamic_shapes(self) -> Self:
        if self.input_specs.keys() != self.input_dynamic_shapes.keys():
            raise ValueError(
                "input_specs and input_dynamic_shapes must contain the same names."
            )
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

    def example_inputs(
        self,
        batch_size: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Tensor]:
        inputs: dict[str, Tensor] = {}
        for name, spec in self.input_specs.items():
            example_batch_size = (
                (
                    batch_size
                    if batch_size is not None
                    else self._default_batch_size(name)
                )
                if spec.is_batched
                else None
            )
            tensor = spec.example_tensor(batch_size=example_batch_size)
            if tensor.is_floating_point():
                tensor = torch.randn(
                    tensor.shape,
                    dtype=dtype if dtype is not None else tensor.dtype,
                    device=device,
                )
            elif device is not None:
                tensor = tensor.to(device=device)
            inputs[name] = tensor
        return inputs

    def dynamic_shapes(
        self, *, dynamic_batch_size: bool = True
    ) -> dict[str, tuple[_DimHint | Dim, ...]]:
        result: dict[str, tuple[_DimHint | Dim, ...]] = {}
        for name, dims in self.input_dynamic_shapes.items():
            new_dims = list(dims)
            if not dynamic_batch_size and self.input_specs[name].is_batched:
                new_dims[0] = Dim.STATIC
            result[name] = tuple(new_dims)
        return result

    @staticmethod
    def _is_static_dim(dim: _DimHint | Dim) -> bool:
        return isinstance(dim, _DimHint) and dim.type.name == "STATIC"

    def _default_batch_size(self, name: str) -> int:
        batch_dim = self.input_dynamic_shapes[name][0]
        minimum = getattr(batch_dim, "min", None)
        return 1 if minimum is None else minimum


@dataclass
class BaseModelOutput(ABC):
    """Base for named model outputs that can cross torch export boundaries."""

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
    values: list[Tensor] = []
    context: list[str] = []
    for field in fields(model_output):
        values.append(getattr(model_output, field.name))
        context.append(field.name)
    return values, context


def _model_output_unflatten(
    values: list[Tensor],
    context: list[str],
    output_type: type[BaseModelOutput],
) -> BaseModelOutput:
    return output_type(**dict(zip(context, values)))
