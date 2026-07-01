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


# TODO (Lionel, 06/26): Remove this test once the DINOv2 LT-DETR models are completely
# migrated to the generic LTDETR pipeline.
def test_dinov2_train_args_v2_matches_original_dinov2_train_args() -> None:
    """DINOv2LTDETRObjectDetectionTrainArgsV2 must replicate the original
    DINOv2LTDETRObjectDetectionTrainArgs defaults exactly.

    The generic LTDETR pipeline reimplements the DINOv2 LTDETR training recipe as a
    separate args class. Regression test to catch accidental drift between the two.
    """
    original = DINOv2LTDETRObjectDetectionTrainArgs()
    v2 = DINOv2LTDETRObjectDetectionTrainArgsV2()

    # patch_size is not present on the original standalone args class: DINOv2 ViT
    # backbones are always patch-14 there, resolved implicitly from the model itself
    # rather than being a configurable train arg.
    v2_dump = v2.model_dump()
    v2_dump.pop("patch_size")
    assert v2_dump == original.model_dump()
    assert (
        DINOv2LTDETRObjectDetectionTrainArgsV2.default_batch_size
        == DINOv2LTDETRObjectDetectionTrainArgs.default_batch_size
    )
    assert (
        DINOv2LTDETRObjectDetectionTrainArgsV2.default_steps
        == DINOv2LTDETRObjectDetectionTrainArgs.default_steps
    )
