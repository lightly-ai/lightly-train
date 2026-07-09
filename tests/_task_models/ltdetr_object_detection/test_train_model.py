#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.dinov2_ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgs,
    DINOv2LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._task_models.ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgsV2,
    DINOv2LTDETRObjectDetectionValTransformArgsV2,
)


# TODO (Lionel, 06/26): Remove this test once the DINOv2 LT-DETR models are completely
# migrated to the generic LTDETR pipeline.
def test_dinov2_transform_args_v2_matches_original_dinov2_transform_args() -> None:
    """DINOv2LTDETRObjectDetection{Train,Val}TransformArgsV2 must replicate the
    original dinov2_ltdetr_object_detection transform args defaults exactly.

    The generic LTDETR pipeline copies the DINOv2 LTDETR transform recipe (in
    particular the patch-14-tuned scale-jitter `sizes` list) verbatim into a
    separate args class rather than deriving it from the generic, patch-16-tuned
    recipe. Regression test to catch accidental drift between the two.
    """
    for v2_args, original_args in (
        (
            DINOv2LTDETRObjectDetectionTrainTransformArgsV2(),
            DINOv2LTDETRObjectDetectionTrainTransformArgs(),
        ),
        (
            DINOv2LTDETRObjectDetectionValTransformArgsV2(),
            DINOv2LTDETRObjectDetectionValTransformArgs(),
        ),
    ):
        # BboxParams (from albumentations) doesn't implement __eq__, so compare its
        # fields via vars() instead of relying on model_dump() equality.
        v2_dump = v2_args.model_dump(exclude={"bbox_params"})
        original_dump = original_args.model_dump(exclude={"bbox_params"})
        assert v2_dump == original_dump
        assert vars(v2_args.bbox_params) == vars(original_args.bbox_params)
