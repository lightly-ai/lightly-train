#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from lightly_train._commands import benchmark_backends
from lightly_train._commands.benchmark_backends import TorchBackend
from lightly_train._commands.benchmark_types import TorchBackendArgs
from lightly_train._pre_post_processing.object_detection import ObjectDetectionOutput
from lightly_train._task_models.task_model import TaskModel


class _FakeModel(TaskModel):
    def __init__(self) -> None:
        super().__init__(
            init_args={"self": self, "__class__": type(self)},
        )
        self.norm = nn.BatchNorm1d(1)
        self.postprocess_inputs: list[Any] = []

    def forward_backend(self, x: Tensor) -> Any:
        return (x.half(), x.half())

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Tensor]]:
        self.postprocess_inputs.append(raw_outputs)
        return [{} for _ in metadata]


class TestTorchBackend:
    def test_init__sets_eval_mode(self) -> None:
        model = _FakeModel().train()

        TorchBackend(
            model=model, backend_args=TorchBackendArgs(), device=torch.device("cpu")
        )

        assert not model.training
        assert not model.norm.training

    def test_run_batch__postprocesses_in_fp32(self) -> None:
        model = _FakeModel()
        backend = TorchBackend(
            model=model, backend_args=TorchBackendArgs(), device=torch.device("cpu")
        )

        backend.run_batch(
            {"image": torch.zeros(2, 3, 8, 8), "original_size": [(8, 8), (8, 8)]}  # type: ignore[typeddict-item]
        )

        (raw_outputs,) = model.postprocess_inputs
        assert all(t.dtype == torch.float32 for t in raw_outputs)


def test_to_float32__casts_tuple() -> None:
    out = benchmark_backends._to_float32(
        (torch.zeros(2, dtype=torch.float16), torch.zeros(2, dtype=torch.long))
    )

    assert isinstance(out, tuple)
    assert out[0].dtype == torch.float32
    assert out[1].dtype == torch.long


def test_to_float32__casts_model_output() -> None:
    out = benchmark_backends._to_float32(
        ObjectDetectionOutput(
            logits=torch.zeros(1, 2, 3, dtype=torch.bfloat16),
            boxes=torch.zeros(1, 2, 4, dtype=torch.float16),
        )
    )

    assert isinstance(out, ObjectDetectionOutput)
    assert out.logits.dtype == torch.float32
    assert out.boxes.dtype == torch.float32
