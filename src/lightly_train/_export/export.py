#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.export import ExportedProgram

from lightly_train._export.onnx_helpers import (
    check_model_input_spec_requirements,
    check_onnx_dynamo_requirements,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import ModelInputSpec


class ExportMixin(ABC):
    """Shared torch.export support for task models with a declared I/O spec."""

    @property
    @abstractmethod
    def model_input_spec(self) -> ModelInputSpec: ...

    def export(
        self,
        *,
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        shape_overrides: dict[str, tuple[int | None, ...]] | None = None,
    ) -> ExportedProgram:
        """Capture the model with the strict torch.export API."""
        check_onnx_dynamo_requirements()
        check_model_input_spec_requirements()
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
        example_inputs = self.model_input_spec.example_inputs(
            batch_size=2 if dynamic_batch_size else batch_size,
            device=device,
            dtype=dtype,
            shape_overrides=shape_overrides,
        )
        return torch.export.export(
            module,
            args=(),
            kwargs=example_inputs,
            dynamic_shapes=self.model_input_spec.dynamic_shapes(
                dynamic_batch_size=dynamic_batch_size
            ),
            strict=True,
        )
