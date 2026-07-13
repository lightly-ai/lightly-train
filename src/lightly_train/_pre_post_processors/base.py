#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Literal,
    cast,
)

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.export import Dim
from torch.export.dynamic_shapes import _DimHint
from torch.nn import Module

from lightly_train._configs.config import PydanticConfig
from lightly_train._export.onnx_helpers import check_onnx_dynamo_requirements

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


class DynamoExportConfig(PydanticConfig):
    out: str | Path
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    batch_size: int | None = None
    dynamic_batch_size: bool = True
    height: int | None = None
    width: int | None = None
    opset_version: int | None = None
    simplify: bool = True
    verify: bool = True
    format_args: dict[str, Any] | None = None


class DynamoExportMixin(ABC):
    """Mixin for modules that should be compiled with full graph optimization and
    exported to ONNX via the dynamo exporter.

    Subclasses must implement ``model_input_spec`` and a ``forward`` that returns a
    ``BaseModelOutput`` subclass. The forward is automatically wrapped in
    ``torch.compile(fullgraph=True)``.
    """

    @property
    @abstractmethod
    def model_input_spec(self) -> ModelInputSpec: ...

    @abstractmethod
    def forward(self, *args: Tensor) -> BaseModelOutput:
        """Interface for processors that can be exported to ONNX.

        The processor should take one or more tensors as input and return a subclass of
        ``BaseModelOutput`` as output.
        """
        ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        original = cls.__dict__["forward"]

        @functools.wraps(original)
        def checked_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            return original(self, *args, **kwargs)

        cls.forward = torch.compile(checked_forward, fullgraph=True)  # type: ignore[method-assign]

    def export_onnx(
        self,
        out: str | Path,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        batch_size: int | None = None,
        dynamic_batch_size: bool = True,
        height: int | None = None,
        width: int | None = None,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> None:
        """Exports the processor to ONNX using the dynamo exporter.

        The export is driven by ``self.model_input_spec``: example inputs, input names
        and dynamic shapes are all derived from the spec, and the output names are
        derived from the ``BaseModelOutput`` returned by ``forward``. Every argument
        defaults to "auto" (``None`` / spec-derived) and can be overridden.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. "auto" keeps the current precision,
                "fp32"/"fp16" cast the model and floating-point inputs accordingly.
            batch_size:
                Batch size used for tracing. If None, defaults to 2 when
                ``dynamic_batch_size`` is True, otherwise 1.
            dynamic_batch_size:
                If True, the batch dimension stays dynamic (as declared in the spec).
                If False, the batch dimension is fixed to ``batch_size``.
            height:
                Overrides the height (second-to-last dim) of every spatial input. If
                None, the spec shape is used.
            width:
                Overrides the width (last dim) of every spatial input. If None, the
                spec shape is used.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to ``torch.onnx.export``.

        Returns:
            None. Writes the ONNX model to ``out``.

        Raises:
            pydantic.ValidationError: If any of the passed arguments are invalid.
        """
        # Validate the passed arguments via the config (strict, forbids extras) and
        # read the validated values back from it.
        config = DynamoExportConfig(
            out=out,
            precision=precision,
            batch_size=batch_size,
            dynamic_batch_size=dynamic_batch_size,
            height=height,
            width=width,
            opset_version=opset_version,
            simplify=simplify,
            verify=verify,
            format_args=format_args,
        )
        out = config.out
        precision = config.precision
        batch_size = config.batch_size
        dynamic_batch_size = config.dynamic_batch_size
        height = config.height
        width = config.width
        opset_version = config.opset_version
        simplify = config.simplify
        verify = config.verify
        format_args = config.format_args

        check_onnx_dynamo_requirements()

        # This mixin is always combined with a torch.nn.Module subclass.
        module = cast(Module, self)
        module.eval()

        first_parameter = next(module.parameters(), None)
        device = (
            first_parameter.device
            if first_parameter is not None
            else torch.device("cpu")
        )

        if precision == "fp32":
            dtype: torch.dtype | None = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision == "auto":
            dtype = None
        else:
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'auto', 'fp32', 'fp16'."
            )

        if dtype is not None:
            module.to(dtype)

        spec = self.model_input_spec
        default_batch_size = 2 if dynamic_batch_size else 1
        batch = batch_size if batch_size is not None else default_batch_size

        # Build the example inputs from the spec, applying overrides where requested.
        example_inputs: dict[str, Tensor] = {}
        for name, tensor_spec in spec.input_specs.items():
            shape = list(tensor_spec.shape)
            if len(shape) >= 2:
                if height is not None:
                    shape[-2] = height
                if width is not None:
                    shape[-1] = width
            if tensor_spec.is_batched:
                shape = [batch, *shape]

            tensor_dtype = tensor_spec.dtype
            if tensor_dtype.is_floating_point:
                if dtype is not None:
                    tensor_dtype = dtype
                example_inputs[name] = torch.randn(
                    shape, dtype=tensor_dtype, device=device
                )
            else:
                example_inputs[name] = torch.zeros(
                    shape, dtype=tensor_dtype, device=device
                )

        # Build the dynamic shapes from the spec, forcing a static batch dim if
        # dynamic batching is disabled.
        dynamic_shapes: dict[str, tuple[_DimHint, ...]] = {}
        for name, dims in spec.input_dynamic_shapes.items():
            new_dims = list(dims)
            if not dynamic_batch_size and spec.input_specs[name].is_batched:
                new_dims[0] = Dim.STATIC
            dynamic_shapes[name] = tuple(new_dims)

        input_names = list(spec.input_specs)

        # Derive the output names from the BaseModelOutput returned by forward.
        with torch.no_grad():
            example_output = module(**example_inputs)
        output_names = output_names_from_model_output(example_output)

        torch.onnx.export(
            module,
            args=(),
            kwargs=example_inputs,
            f=str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            **(format_args or {}),
        )

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                model=str(out),
                output_model=str(out),
                # We skip constant folding as this currently increases the model size by
                # quite a lot.
                skip_optimizations=["constant_folding"],
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            from lightly_train import _torch_testing

            onnx.checker.check_model(str(out), full_check=True)

            # Always run the reference input in float32 and on cpu for consistency.
            reference_model = copy.deepcopy(module).cpu().to(torch.float32).eval()
            reference_inputs = {
                name: (
                    tensor.detach().cpu().to(torch.float32)
                    if tensor.is_floating_point()
                    else tensor.detach().cpu()
                )
                for name, tensor in example_inputs.items()
            }
            reference_output = reference_model(**reference_inputs)
            reference_values, _ = _model_output_flatten(reference_output)

            # Get outputs from the ONNX model.
            session = ort.InferenceSession(str(out))
            input_feed = {
                name: tensor.detach().cpu().numpy()
                for name, tensor in example_inputs.items()
            }
            outputs_onnx = session.run(output_names=None, input_feed=input_feed)
            outputs_onnx = tuple(torch.from_numpy(y) for y in outputs_onnx)

            if len(outputs_onnx) != len(reference_values):
                raise AssertionError(
                    f"Number of onnx outputs should be {len(reference_values)} but is "
                    f"{len(outputs_onnx)}"
                )
            for output_onnx, output_model, output_name in zip(
                outputs_onnx, reference_values, output_names
            ):

                def msg(s: str) -> str:
                    return f'ONNX validation failed for output "{output_name}": {s}'

                if output_model.is_floating_point():
                    # Absolute and relative tolerances are a bit arbitrary and taken from:
                    # https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_core.py#L1611-L1618
                    torch.testing.assert_close(
                        output_onnx,
                        output_model,
                        msg=msg,
                        equal_nan=True,
                        check_device=False,
                        check_dtype=False,
                        check_layout=False,
                        atol=5e-3,
                        rtol=1e-1,
                    )
                else:
                    _torch_testing.assert_most_equal(
                        output_onnx,
                        output_model,
                        msg=msg,
                    )

        logger.info(f"Successfully exported ONNX model to '{out}'")
