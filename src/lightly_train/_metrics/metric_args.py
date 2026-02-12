#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from torchmetrics import Metric

from lightly_train._configs.config import PydanticConfig


class MetricArgs(PydanticConfig):
    """Base class for individual metric arguments."""

    def get_metrics(
        self, *, classwise: bool = False, num_classes: int
    ) -> dict[str, Metric]:
        """Create metric instances.

        Args:
            classwise: If True, return metrics with average='none' for classwise
                      computation. The ClasswiseWrapper will be applied by the
                      TaskMetric class.
            num_classes: Number of classes for the classification task.

        Returns:
            Dictionary mapping metric names to metric instances.
            Example: {"top1_acc_micro": MulticlassAccuracy(...), "top5_acc_micro": ...}
        """
        raise NotImplementedError

    def supports_classwise(self) -> bool:
        """Whether this metric supports classwise computation.

        Returns False for metrics like topk>1 accuracy that don't make sense classwise.
        """
        return True
