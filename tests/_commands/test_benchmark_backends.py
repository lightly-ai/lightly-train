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

from lightly_train._commands.benchmark_backends import (
    ObjectDetectionPipeline,
    TorchBackend,
)
from lightly_train._commands.benchmark_types import TorchBackendArgs
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    ObjectDetectionMetadata,
    ObjectDetectionPostprocessor,
    ObjectDetectionPrediction,
    ObjectDetectionPreprocessor,
)
from lightly_train._task_models.task_model import TaskModel


class _RecordingPostprocessor(ObjectDetectionPostprocessor):
    def __init__(self) -> None:
        super().__init__(num_top_queries=1, internal_class_to_class=torch.tensor([0]))
        self.raw_inputs: list[ObjectDetectionBatchOutput] = []

    def postprocess(  # type: ignore[override]
        self,
        raw: ObjectDetectionBatchOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        self.raw_inputs.append(raw)
        return []


class _FakeModel(TaskModel):
    def __init__(self) -> None:
        super().__init__(init_args={"self": self, "__class__": type(self)})
        self.norm = nn.BatchNorm1d(1)

    def forward(self, images: Tensor) -> Any:
        return ObjectDetectionBatchOutput(
            logits=torch.zeros(images.shape[0], 1, 1), boxes=torch.zeros(1, 1, 4)
        )


def _preprocessor() -> ObjectDetectionPreprocessor:
    return ObjectDetectionPreprocessor(
        image_size=(8, 8), image_normalize=None, expected_input_channels=3
    )


class TestObjectDetectionPipeline:
    def test_postprocess__decodes_in_fp32(self) -> None:
        postprocessor = _RecordingPostprocessor()
        pipeline = ObjectDetectionPipeline(
            preprocessor=_preprocessor(),
            postprocessor=postprocessor,
            device=torch.device("cpu"),
        )

        pipeline.postprocess(
            ObjectDetectionBatchOutput(
                logits=torch.zeros(1, 2, 3, dtype=torch.bfloat16),
                boxes=torch.zeros(1, 2, 4, dtype=torch.float16),
            ),
            metadata=[ObjectDetectionMetadata(orig_h=8, orig_w=8)],
        )

        (raw,) = postprocessor.raw_inputs
        assert raw.logits.dtype == torch.float32
        assert raw.boxes.dtype == torch.float32


class TestTorchBackend:
    def test_init__sets_eval_mode(self) -> None:
        model = _FakeModel().train()

        TorchBackend(
            model=model,
            backend_args=TorchBackendArgs(),
            device=torch.device("cpu"),
            preprocessor=_preprocessor(),
            postprocessor=_RecordingPostprocessor(),
        )

        assert not model.training
        assert not model.norm.training
