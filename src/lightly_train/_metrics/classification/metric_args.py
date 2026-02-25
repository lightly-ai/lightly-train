#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torchmetrics import Metric

from lightly_train._metrics.metric_args import MetricArgs


class ClassificationMetricArgs(MetricArgs):
    def get_metrics(self, *, classwise: bool, num_classes: int) -> dict[str, Metric]:
        raise NotImplementedError("Must be implemented by subclasses.")
