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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any, List, Union, cast, overload

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.export.dynamic_shapes import Dim
from typing_extensions import Self, TypeAlias

from lightly_train._export.onnx_helpers import check_model_input_spec_requirements


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
    input_dynamic_shapes: dict[str, tuple[Any, ...]] = Field(
        ...,
        description=(
            "Dynamic shapes of the model's named inputs, should be either of type "
            "torch.export.dynamic_shapes.Dim or torch.export.dynamic_shapes._DimHint."
        ),
    )

    @model_validator(mode="after")
    def _validate_input_dynamic_shapes(self) -> Self:
        """Validate names, ranks, and allowed dynamic dimensions of model inputs.

        Every input must have a matching dynamic-shape entry whose rank includes
        the batch dimension for batched inputs. Only that leading batch dimension
        may be dynamic; all remaining dimensions must be static.
        """
        check_model_input_spec_requirements()
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
        shape_overrides: dict[str, tuple[int | None, ...]] | None = None,
    ) -> dict[str, Tensor]:
        """Create example inputs, optionally overriding static input dimensions.

        ``shape_overrides`` excludes the batch dimension. A value of ``None`` keeps
        the corresponding dimension declared in ``input_specs``.
        """
        shape_overrides = shape_overrides or {}
        unknown_inputs = shape_overrides.keys() - self.input_specs.keys()
        if unknown_inputs:
            raise ValueError(
                "shape_overrides contains unknown input names: "
                f"{sorted(unknown_inputs)}."
            )

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
            shape = self._resolve_shape(
                name=name, shape=spec.shape, override=shape_overrides.get(name)
            )
            tensor_spec = spec.model_copy(update={"shape": shape})
            tensor = tensor_spec.example_tensor(batch_size=example_batch_size)
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
    ) -> dict[str, tuple[Dim | Any, ...]]:
        result: dict[str, tuple[Dim | Any, ...]] = {}
        for name, dims in self.input_dynamic_shapes.items():
            new_dims = list(dims)
            if not dynamic_batch_size and self.input_specs[name].is_batched:
                new_dims[0] = Dim.STATIC
            result[name] = tuple(new_dims)
        return result

    @staticmethod
    def _is_static_dim(dim: Dim | Any) -> bool:
        return bool(dim == Dim.STATIC)

    def _default_batch_size(self, name: str) -> int:
        batch_dim = self.input_dynamic_shapes[name][0]
        minimum = getattr(batch_dim, "min", None)
        return 1 if minimum is None else minimum

    @staticmethod
    def _resolve_shape(
        *,
        name: str,
        shape: tuple[int, ...],
        override: tuple[int | None, ...] | None,
    ) -> tuple[int, ...]:
        if override is None:
            return shape
        if len(override) != len(shape):
            raise ValueError(
                f"shape_overrides for '{name}' has rank {len(override)}, "
                f"expected {len(shape)}."
            )
        if any(dimension is not None and dimension <= 0 for dimension in override):
            raise ValueError(
                f"shape_overrides for '{name}' must contain only positive dimensions "
                "or None."
            )
        return tuple(
            default if override_dimension is None else override_dimension
            for default, override_dimension in zip(shape, override)
        )


@functools.lru_cache(maxsize=None)
def _field_names(output_type: type) -> tuple[str, ...]:
    """Return the dataclass field names of an output type, cached per class."""
    return tuple(field.name for field in fields(output_type))


@dataclass
class BaseModelOutput(Mapping[str, Tensor], ABC):
    """Base for named model outputs that can cross torch export boundaries.

    Behaves like a read-only ``Mapping[str, Tensor]`` (supports ``[]``, ``in``,
    ``.keys()``/``.items()``/``.values()``/``.get()``, ``dict(x)``, and ``**x``
    unpacking) in addition to attribute access, for backward compatibility with
    code written against dict-based prediction outputs. Note that
    ``isinstance(x, dict)`` is NOT true for these objects — check
    ``isinstance(x, collections.abc.Mapping)`` instead.

    Mutating dict methods (``x[key] = ...``, ``.update()``, ``.pop()``, ``.copy()``)
    are NOT supported. Use ``dict(x)`` to get a mutable copy.

    Outputs whose fields share a leading row dimension should subclass
    :class:`RowIndexableOutput` instead, which adds row filtering on top.
    """

    def __getitem__(self, key: str) -> Tensor:
        """Return a declared output field by name."""
        if key not in _field_names(type(self)):
            raise KeyError(key)
        return cast(Tensor, getattr(self, key))

    def __iter__(self) -> Iterator[str]:
        return iter(_field_names(type(self)))

    def __len__(self) -> int:
        return len(_field_names(type(self)))

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


RowIndex: TypeAlias = Union[Tensor, int, slice, List[int]]


@dataclass
class RowIndexableOutput(BaseModelOutput, ABC):
    """A model output whose fields all share a leading row dimension ``N``.

    Adds row filtering on top of the ``Mapping`` behaviour of
    :class:`BaseModelOutput`::

        kept = prediction[prediction.scores > 0.5]
        cats = prediction[prediction.labels == 17]

    Indexing with a ``str`` keeps the ``Mapping`` semantics and returns the named
    field. Indexing with anything else selects rows and returns a NEW instance of the
    same class whose tensors never alias the original ones.

    Note:
        ``len(output)`` and iterating over an output keep their ``Mapping`` meaning:
        the number of fields and the field names, NOT the number of rows. Use
        :attr:`num_rows` to count rows.
    """

    @property
    def num_rows(self) -> int:
        """Number of rows ``N`` shared by all fields."""
        return int(self._first_field().shape[0])

    def select(self, index: RowIndex) -> Self:
        """Return a new output containing only the selected rows.

        Args:
            index:
                A 1-dimensional boolean mask of length :attr:`num_rows`, a
                1-dimensional integer tensor, a list of integers, a slice, or a single
                integer (which keeps the row dimension).

        Returns:
            A new instance of the same class. Its tensors are always copies, never
            views of the original tensors.
        """
        normalized = self._normalize_index(index)
        values = {
            name: cast(Tensor, getattr(self, name))[normalized.to(self._device(name))]
            for name in _field_names(type(self))
        }
        return type(self)(**values)

    @overload
    def __getitem__(self, index: str) -> Tensor: ...

    @overload
    def __getitem__(self, index: RowIndex) -> Self: ...

    def __getitem__(self, index: Union[str, RowIndex]) -> Union[Tensor, Self]:
        """Return a field by name, or a new output with the selected rows."""
        if isinstance(index, str):
            return super().__getitem__(index)
        return self.select(index)

    def __contains__(self, key: object) -> bool:
        # Mapping.__contains__ delegates to __getitem__ and would therefore report a
        # successful row selection as a successful field lookup.
        return isinstance(key, str) and key in _field_names(type(self))

    def _first_field(self) -> Tensor:
        return cast(Tensor, getattr(self, _field_names(type(self))[0]))

    def _device(self, name: str) -> torch.device:
        return cast(Tensor, getattr(self, name)).device

    def _normalize_index(self, index: RowIndex) -> Tensor:
        """Return an index that is guaranteed to copy rather than return a view."""
        device = self._first_field().device
        if isinstance(index, Tensor):
            if index.ndim != 1:
                raise IndexError(
                    f"Index tensor must be 1-dimensional but has {index.ndim} "
                    "dimensions."
                )
            if index.dtype == torch.bool:
                if index.shape[0] != self.num_rows:
                    raise IndexError(
                        f"Boolean mask has length {index.shape[0]} but the output has "
                        f"{self.num_rows} rows."
                    )
                return index
            if index.dtype not in (torch.int32, torch.int64):
                raise IndexError(
                    "Index tensor must have dtype bool, int32, or int64 but has "
                    f"{index.dtype}."
                )
            return index.to(dtype=torch.int64)
        # bool is a subclass of int, so it must be rejected before the int branch.
        if isinstance(index, bool):
            raise TypeError(
                "Indexing with a bool is not supported, use a 1-dimensional bool "
                "tensor mask instead."
            )
        if isinstance(index, int):
            rows = [index]
        elif isinstance(index, slice):
            rows = list(range(*index.indices(self.num_rows)))
        else:
            rows = list(index)
        return torch.tensor(rows, dtype=torch.int64, device=device)
