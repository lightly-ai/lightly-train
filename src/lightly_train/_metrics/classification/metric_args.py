#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torchmetrics import Metric as TorchmetricsMetric

from lightly_train._metrics.metric_args import MetricArgs


class ClassificationMetricArgs(MetricArgs):
    def get_torchmetrics_instances(
        self, *, classwise: bool, num_classes: int
    ) -> dict[str, TorchmetricsMetric]:
        raise NotImplementedError("Must be implemented by subclasses.")
