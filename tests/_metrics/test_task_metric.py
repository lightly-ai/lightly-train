#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import pytest
from lightning_utilities.core.imports import RequirementCache

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)


from lightly_train._metrics.classification.task_metric import (
    MultilabelClassificationTaskMetricArgs,
)
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetricArgs,
)
from lightly_train._metrics.task_metric import get_watch_metric_mode


class TestGetWatchMetricMode:
    @pytest.mark.parametrize(
        "watch_metric, loss_names, expected",
        [
            ("val_metric/miou", ["loss"], "max"),
            ("val_loss", ["loss"], "min"),
            ("val_loss/loss_vfl", ["loss", "loss_vfl"], "min"),
            # Classwise and multihead variants use substring matching.
            ("val_metric_classwise/iou_car", ["loss"], "max"),
            ("val_metric_head/miou_lr0_001", ["loss"], "max"),
        ],
    )
    def test_get_watch_metric_mode(
        self, watch_metric: str, loss_names: list[str], expected: str
    ) -> None:
        args = SemanticSegmentationTaskMetricArgs()
        assert get_watch_metric_mode(args, loss_names, watch_metric) == expected

    def test_get_watch_metric_mode__min_metric(self) -> None:
        # HammingDistance has watch_mode="min".
        args = MultilabelClassificationTaskMetricArgs()
        assert (
            get_watch_metric_mode(args, ["loss"], "val_metric/hamming_distance")
            == "min"
        )

    def test_get_watch_metric_mode__not_found(self) -> None:
        args = SemanticSegmentationTaskMetricArgs()
        with pytest.raises(ValueError):
            get_watch_metric_mode(args, ["loss"], "val_metric/unknown_metric")
