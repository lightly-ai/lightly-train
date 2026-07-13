#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import json
import logging
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Module

from lightly_train._configs.config import PydanticConfig
from lightly_train._export.export import ExportMixin
from lightly_train._export.onnx_helpers import (
    fix_topological_order,
    remove_redundant_casts,
    write_onnx_metadata,
)
from lightly_train._task_models import task_model_io
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import BaseModelOutput

logger = logging.getLogger(__name__)


class DynamoExportConfig(PydanticConfig):
    out: str | Path
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    batch_size: int | None = None
    dynamic_batch_size: bool = True
    opset_version: int | None = None
    simplify: bool = True
    verify: bool = True
    format_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class ONNXExportPrecisionPolicy:
    """Precision policy for ONNX export.

    Args:
        fp32_module_names:
            Fully-qualified module names or module name prefixes to keep in FP32 after
            casting the model to the requested export precision.
        fp32_module_types:
            Module classes to keep in FP32 after casting the model to the requested
            export precision.
        fp32_onnx_op_types:
            ONNX operator types to keep in FP32 when converting an exported graph to
            FP16.
    """

    fp32_module_names: tuple[str, ...] = ()
    fp32_module_types: tuple[type[Module], ...] = ()
    fp32_onnx_op_types: tuple[str, ...] = ()


class ONNXExportMixin(ExportMixin):
    @property
    def onnx_export_precision_policy(self) -> ONNXExportPrecisionPolicy:
        return ONNXExportPrecisionPolicy()

    @abstractmethod
    def verify_onnx_export_outputs(
        self,
        *,
        torch_outputs: BaseModelOutput,
        onnx_outputs: BaseModelOutput,
    ) -> None: ...

    def onnx_export_metadata(self) -> dict[str, str]:
        """String metadata embedded into the exported ONNX (``metadata_props``).

        This mirrors the information stored in training checkpoints so the ONNX
        model is self-describing. Override to add or customize entries.

        ``image_size`` is intentionally omitted: it is already readable from the
        static height/width of the ONNX input tensor's shape.
        """
        from lightly_train import __version__  # lazy: avoid import cycle
        from lightly_train._license import LICENSE_INFO

        metadata: dict[str, str] = {
            "lightly_train_version": __version__,
            "license_info": LICENSE_INFO,
        }
        image_normalize = getattr(self, "image_normalize", None)
        if image_normalize is not None:
            metadata["image_normalize"] = json.dumps(image_normalize, sort_keys=True)
        classes = getattr(self, "classes", None)
        if classes is not None:
            metadata["classes"] = json.dumps(classes, sort_keys=True)
        model_name = getattr(self, "init_args", {}).get("model_name")
        if model_name is not None:
            metadata["model_name"] = str(model_name)
        return metadata

    def export_onnx(
        self,
        out: str | Path,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        batch_size: int | None = None,
        dynamic_batch_size: bool = True,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> None:
        """Exports the processor to ONNX using the dynamo exporter.

        The export is driven by ``self.model_input_spec``: example inputs, input names
        and dynamic shapes are all derived from the spec, and the output names are
        derived from the ``BaseModelOutput`` returned by ``forward``.

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
        # Validate the passed arguments via the config (strict, forbids extras).
        config = DynamoExportConfig(
            out=out,
            precision=precision,
            batch_size=batch_size,
            dynamic_batch_size=dynamic_batch_size,
            opset_version=opset_version,
            simplify=simplify,
            verify=verify,
            format_args=format_args,
        )

        if not isinstance(self, TaskModel):
            raise TypeError("ExportMixin can only be used with TaskModel subclasses.")

        module: TaskModel = self
        module.eval()
        module.deploy()

        # Cast the module to the requested precision before tracing.
        if config.precision == "fp32":
            precision_dtype: torch.dtype | None = torch.float32
        elif config.precision == "fp16":
            precision_dtype = torch.float16
        else:  # "auto"
            precision_dtype = None
        self._apply_onnx_export_module_precision(module=module, dtype=precision_dtype)

        program = self.export(
            batch_size=config.batch_size,
            dynamic_batch_size=config.dynamic_batch_size,
        )

        # Derive the ONNX naming and verification inputs from the model input spec,
        # matching the module's current (post-cast) dtype/device.
        first_parameter = next(module.parameters(), None)
        device = (
            first_parameter.device
            if first_parameter is not None
            else torch.device("cpu")
        )
        dtype = first_parameter.dtype if first_parameter is not None else None

        spec = self.model_input_spec
        batch = (
            config.batch_size
            if config.batch_size is not None
            else (2 if config.dynamic_batch_size else 1)
        )
        example_inputs = spec.example_inputs(
            batch_size=batch, device=device, dtype=dtype
        )
        input_names = list(spec.input_specs)
        dynamic_shapes = spec.dynamic_shapes(
            dynamic_batch_size=config.dynamic_batch_size
        )

        with torch.no_grad():
            example_output = module(**example_inputs)
        output_names = [field.name for field in fields(example_output)]

        # When passing an ExportedProgram, torch ignores args/kwargs for tracing
        # (already baked into the program) but still applies the input/output names
        # and uses ``dynamic_shapes`` to name the dynamic ONNX axes.
        torch.onnx.export(
            program,
            f=str(config.out),
            input_names=input_names,
            output_names=output_names,
            opset_version=config.opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            **(config.format_args or {}),
        )

        self._apply_onnx_export_graph_precision(
            out=config.out,
            precision=config.precision,
        )

        if config.simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                model=str(config.out),
                output_model=str(config.out),
                # We skip constant folding as this currently increases the model size by
                # quite a lot.
                skip_optimizations=["constant_folding"],
            )

        # Write metadata last: the graph-precision conversion and onnxslim both
        # rewrite the file and can drop metadata_props, so this must run after them.
        write_onnx_metadata(out=config.out, metadata=self.onnx_export_metadata())

        if config.verify:
            self._verify_onnx_export(
                out=config.out,
                module=module,
                example_inputs=example_inputs,
            )

        logger.info(f"Successfully exported ONNX model to '{config.out}'")

    def _apply_onnx_export_module_precision(
        self,
        *,
        module: Module,
        dtype: torch.dtype | None,
    ) -> None:
        """Apply the requested module precision and FP32 module overrides."""
        if dtype is not None:
            module.to(dtype)
        if dtype != torch.float16:
            return
        policy = self.onnx_export_precision_policy
        for name, child_module in module.named_modules():
            if self._is_onnx_export_fp32_module_name(
                name=name,
                fp32_module_names=policy.fp32_module_names,
            ) or isinstance(child_module, policy.fp32_module_types):
                child_module.to(torch.float32)

    @staticmethod
    def _is_onnx_export_fp32_module_name(
        *,
        name: str,
        fp32_module_names: tuple[str, ...],
    ) -> bool:
        """Return whether ``name`` is a configured FP32 module or descendant."""
        return any(
            name == fp32_name or name.startswith(f"{fp32_name}.")
            for fp32_name in fp32_module_names
        )

    def _apply_onnx_export_graph_precision(
        self,
        *,
        out: str | Path,
        precision: Literal["auto", "fp32", "fp16"],
    ) -> None:
        """Apply ONNX graph precision policy after export."""
        policy = self.onnx_export_precision_policy
        if precision != "fp16" or not policy.fp32_onnx_op_types:
            return

        import onnx
        from onnxruntime.transformers import float16 as ort_float16

        model_onnx = onnx.load(str(out))
        op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + list(
            policy.fp32_onnx_op_types
        )
        model_fp16 = ort_float16.convert_float_to_float16(
            model_onnx,
            op_block_list=op_block_list,
        )
        remove_redundant_casts(model_fp16)
        fix_topological_order(model_fp16)
        onnx.save(model_fp16, str(out))

    def _verify_onnx_export(
        self,
        *,
        out: str | Path,
        module: Module,
        example_inputs: dict[str, Tensor],
    ) -> None:
        """Validate the ONNX file and compare ONNX Runtime outputs to PyTorch."""
        logger.info("Verifying ONNX model")
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(out), full_check=True)

        # Always run the reference input in float32 and on cpu for consistency.
        reference_model = copy.deepcopy(module).cpu().to(torch.float32).eval()
        if isinstance(reference_model, TaskModel):
            reference_model.deploy()
        reference_inputs = {
            name: (
                tensor.detach().cpu().to(torch.float32)
                if tensor.is_floating_point()
                else tensor.detach().cpu()
            )
            for name, tensor in example_inputs.items()
        }
        reference_output: BaseModelOutput = reference_model(**reference_inputs)
        reference_values, context = task_model_io._model_output_flatten(
            reference_output
        )

        # Get outputs from the ONNX model.
        session = ort.InferenceSession(str(out))
        get_session_inputs: Callable[[], list[Any]] = getattr(
            session, "get_inputs", lambda: []
        )
        session_input_types = {
            input_.name: input_.type for input_ in get_session_inputs()
        }
        input_feed = {}
        for name, tensor in example_inputs.items():
            tensor = tensor.detach().cpu()
            if tensor.is_floating_point() and session_input_types.get(name) == (
                "tensor(float16)"
            ):
                tensor = tensor.half()
            input_feed[name] = tensor.numpy()
        outputs_onnx = session.run(output_names=None, input_feed=input_feed)
        onnx_values = [torch.from_numpy(y) for y in outputs_onnx]

        if len(onnx_values) != len(reference_values):
            raise AssertionError(
                f"Number of onnx outputs should be {len(reference_values)} but is "
                f"{len(onnx_values)}"
            )
        onnx_output = task_model_io._model_output_unflatten(
            onnx_values,
            context,
            output_type=type(reference_output),
        )
        self.verify_onnx_export_outputs(
            torch_outputs=reference_output,
            onnx_outputs=onnx_output,
        )
