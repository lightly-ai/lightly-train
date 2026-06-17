#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import (
    Protocol,
    runtime_checkable,
)

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.export.dynamic_shapes import _DimHint


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

    def example_tensor(self) -> Tensor:
        """Generate an example tensor based on the specified shape and dtype.

        Returns:
            A tensor with the defined shape and data type.
        """
        return torch.randn(self.shape, dtype=self.dtype)


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


class ModelOutputSpec(BaseModel):
    output_names: list[str] = Field(
        ..., description="List of output names for the model."
    )


@dataclass
class BaseModelOutput(ABC):
    def __init_subclass__(cls, **kwargs):
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


class FullGraphMixin(ABC):
    """A base class for modules that should be compiled with full graph optimization."""

    @property
    @abstractmethod
    def model_input_spec(self) -> ModelInputSpec: ...

    @property
    @abstractmethod
    def model_output_spec(self) -> ModelOutputSpec: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original = cls.__dict__["forward"]

        @functools.wraps(original)
        def checked_forward(self, *args, **kwargs):
            return original(self, *args, **kwargs)

        cls.forward = torch.compile(checked_forward, fullgraph=True)


@runtime_checkable
class ExportableProcessor(Protocol):
    def forward(self, *args: Tensor) -> BaseModelOutput:
        """Defines the interface for post processors that can be exported to ONNX.

        The post processor should take one or more tensors as input and return a
        subclass of BaseModelOutput as output.
        """
        ...
