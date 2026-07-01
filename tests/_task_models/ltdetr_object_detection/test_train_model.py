#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.dinov2_ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrainArgsV2,
)


def test_dinov2_train_args_v2_matches_original_dinov2_train_args() -> None:
    """DINOv2LTDETRObjectDetectionTrainArgsV2 must replicate the original
    DINOv2LTDETRObjectDetectionTrainArgs defaults exactly.

    The generic LTDETR pipeline reimplements the DINOv2 LTDETR training recipe as a
    separate args class. Regression test to catch accidental drift between the two.
    """
    original = DINOv2LTDETRObjectDetectionTrainArgs()
    v2 = DINOv2LTDETRObjectDetectionTrainArgsV2()

    assert v2.model_dump() == original.model_dump()
    assert (
        DINOv2LTDETRObjectDetectionTrainArgsV2.default_batch_size
        == DINOv2LTDETRObjectDetectionTrainArgs.default_batch_size
    )
    assert (
        DINOv2LTDETRObjectDetectionTrainArgsV2.default_steps
        == DINOv2LTDETRObjectDetectionTrainArgs.default_steps
    )
