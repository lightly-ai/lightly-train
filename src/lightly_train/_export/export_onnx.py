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
from dataclasses import dataclass, fields
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module

from lightly_train import _logging
from lightly_train._commands import _warnings
from lightly_train._export.export import ExportMixin
from lightly_train._export.onnx_helpers import (
    fix_topological_order,
    remove_duplicate_cast_nodes,
    remove_redundant_casts,
    write_onnx_metadata,
)
from lightly_train._task_models import task_model_io
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import BaseModelOutput
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ONNXExportPrecisionPolicy:
    """Task-specific operators that must stay in FP32 in FP16 ONNX exports."""

    fp32_onnx_op_types: tuple[str, ...] = ()


class ONNXExportMixin(ExportMixin):
    """Common ONNX export pipeline for TaskModels with typed I/O."""

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
        """Return standard metadata embedded in the exported ONNX model."""
        from lightly_train import __version__
        from lightly_train._license import LICENSE_INFO

        metadata = {
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
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, object] | None = None,
        shape_overrides: dict[str, tuple[int | None, ...]] | None = None,
    ) -> None:
        """Export the model to ONNX from its ``ModelInputSpec``."""
        _warnings.filter_export_warnings()
        _logging.set_up_console_logging()
        if not isinstance(self, TaskModel):
            raise TypeError("ONNXExportMixin can only be used with TaskModel subclasses.")
        if precision == "fp16" and not simplify:
            raise ValueError("fp16 precision requires simplify=True.")

        module: TaskModel = self
        module.eval()
        # Trace in FP32: LT-DETR contains FP32 decoder regions and converts to FP16
        # safely after export. The shared pipeline deliberately preserves that behavior.
        module.to(torch.float32)
        module.deploy()

        program = self.export(
            batch_size=batch_size,
            dynamic_batch_size=dynamic_batch_size,
            shape_overrides=shape_overrides,
        )
        first_parameter = next(module.parameters(), None)
        device = (
            first_parameter.device if first_parameter is not None else torch.device("cpu")
        )
        example_inputs = self.model_input_spec.example_inputs(
            batch_size=2 if dynamic_batch_size else batch_size,
            device=device,
            dtype=torch.float32,
            shape_overrides=shape_overrides,
        )
        dynamic_shapes = self.model_input_spec.dynamic_shapes(
            dynamic_batch_size=dynamic_batch_size
        )
        with torch.no_grad():
            example_output = module(**example_inputs)
        output_names = [field.name for field in fields(example_output)]

        logger.info(f"Exporting ONNX model to '{out}'")
        torch.onnx.export(
            program,
            f=str(out),
            input_names=list(self.model_input_spec.input_specs),
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            report=False,
            optimize=True,
            verify=True,
            dynamic_shapes=dynamic_shapes,
            **(format_args or {}),
        )
        self._apply_onnx_export_graph_precision(out=out, precision=precision)

        if simplify:
            import onnxslim  # type: ignore[import-not-found,import-untyped]

            onnxslim.slim(
                str(out), output_model=out, skip_optimizations=["constant_folding"]
            )
        write_onnx_metadata(out=out, metadata=self.onnx_export_metadata())

        if verify:
            self._verify_onnx_export(
                out=out,
                module=module,
                example_inputs=example_inputs,
                precision=precision,
            )
        logger.info(f"Successfully exported ONNX model to '{out}'")

    def _apply_onnx_export_graph_precision(
        self, *, out: PathLike, precision: Literal["fp32", "fp16"]
    ) -> None:
        if precision != "fp16":
            return
        import onnx
        from onnxruntime.transformers import float16 as ort_float16

        model = onnx.load(str(out))
        op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + list(
            self.onnx_export_precision_policy.fp32_onnx_op_types
        )
        model = ort_float16.convert_float_to_float16(model, op_block_list=op_block_list)
        remove_duplicate_cast_nodes(model)
        remove_redundant_casts(model)
        fix_topological_order(model)
        onnx.save(model, str(out))

    def _verify_onnx_export(
        self,
        *,
        out: PathLike,
        module: Module,
        example_inputs: dict[str, Tensor],
        precision: Literal["fp32", "fp16"],
    ) -> None:
        logger.info("Verifying ONNX model")
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(out), full_check=True)
        if precision == "fp16" and "CUDAExecutionProvider" not in ort.get_available_providers():
            logger.warning(
                "Skipping ONNX runtime verification for fp16 model because "
                "CUDAExecutionProvider is not available in onnxruntime."
            )
            return

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
        reference_output = reference_model(**reference_inputs)
        reference_values, context = task_model_io._model_output_flatten(reference_output)

        session = ort.InferenceSession(str(out))
        session_input_types = {
            input_.name: input_.type for input_ in session.get_inputs()
        }
        input_feed = {}
        for name, tensor in example_inputs.items():
            tensor = tensor.detach().cpu()
            if tensor.is_floating_point() and session_input_types.get(name) == "tensor(float16)":
                tensor = tensor.half()
            input_feed[name] = tensor.numpy()
        onnx_values = [torch.from_numpy(value) for value in session.run(None, input_feed)]
        if len(onnx_values) != len(reference_values):
            raise AssertionError(
                f"Number of ONNX outputs should be {len(reference_values)} but is "
                f"{len(onnx_values)}"
            )
        onnx_output = task_model_io._model_output_unflatten(
            onnx_values, context, output_type=type(reference_output)
        )
        self.verify_onnx_export_outputs(
            torch_outputs=reference_output, onnx_outputs=onnx_output
        )
