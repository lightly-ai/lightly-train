#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from torch.export import ExportedProgram

from lightly_train._export.onnx_helpers import check_onnx_dynamo_requirements
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import ModelInputSpec

logger = logging.getLogger(__name__)


class ExportMixin(ABC):
    @property
    @abstractmethod
    def model_input_spec(self) -> ModelInputSpec: ...

    def export(
        self,
        *,
        batch_size: int | None = None,
        dynamic_batch_size: bool = True,
    ) -> ExportedProgram:
        """Trace the model into a ``torch.export.ExportedProgram``.

        Example inputs and dynamic shapes are derived from ``self.model_input_spec``.
        The model is traced in its current precision and configuration; cast or
        deploy it beforehand (as ``export_onnx`` does) to change those.

        Args:
            batch_size:
                Batch size used for tracing. If None, defaults to 2 when
                ``dynamic_batch_size`` is True, otherwise 1.
            dynamic_batch_size:
                If True, the batch dimension stays dynamic (as declared in the spec).
                If False, the batch dimension is fixed to ``batch_size``.

        Returns:
            The traced ``ExportedProgram``.
        """
        check_onnx_dynamo_requirements()

        if not isinstance(self, TaskModel):
            raise TypeError("ExportMixin can only be used with TaskModel subclasses.")

        module: TaskModel = self
        module.eval()

        first_parameter = next(module.parameters(), None)
        device = (
            first_parameter.device
            if first_parameter is not None
            else torch.device("cpu")
        )
        dtype = first_parameter.dtype if first_parameter is not None else None

        spec = self.model_input_spec
        batch = (
            batch_size if batch_size is not None else (2 if dynamic_batch_size else 1)
        )
        example_inputs = spec.example_inputs(
            batch_size=batch, device=device, dtype=dtype
        )
        dynamic_shapes = spec.dynamic_shapes(dynamic_batch_size=dynamic_batch_size)

        # Capture the ExportedProgram using the same fallback chain as torch.onnx's
        # dynamo exporter (non-strict -> strict -> draft_export). Each strategy
        # handles dynamic (and 0/1) dimensions and refines the dynamic shapes from
        # torch's suggested fixes, so this matches the behavior of the previous
        # torch.onnx.export(..., dynamo=True) call.
        from torch.onnx._internal.exporter import _capture_strategies

        program = None
        first_exception: Exception | None = None
        with torch.no_grad():
            for strategy_class in _capture_strategies.CAPTURE_STRATEGIES:
                result = strategy_class(verbose=False)(
                    module, (), example_inputs, dynamic_shapes=dynamic_shapes
                )
                if result.success:
                    program = result.exported_program
                    break
                if first_exception is None:
                    first_exception = result.exception

        if program is None:
            assert first_exception is not None
            raise first_exception
        return program
