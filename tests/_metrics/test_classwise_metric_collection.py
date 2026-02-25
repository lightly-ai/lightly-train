#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision

from lightly_train._metrics.classwise_metric_collection import (
    ClasswiseMetricCollection,
)


class TestClasswiseMetricCollection:
    def test_classwise_metric_collection(self) -> None:
        metrics = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=2, average="none"),
                "precision": MulticlassPrecision(num_classes=2, average="none"),
            },
            prefix="prefix1/",
        )
        metric = ClasswiseMetricCollection(
            metrics=metrics,
            class_names=["cat", "dog"],
            prefix="prefix2/",
            classwise_prefix="classprefix",
        )

        preds = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        target = torch.tensor([0, 1])
        metric.update(preds, target)
        result = metric.compute()
        assert result.keys() == {
            "prefix2/prefix1/accuracy_classprefix_cat",
            "prefix2/prefix1/accuracy_classprefix_dog",
            "prefix2/prefix1/precision_classprefix_cat",
            "prefix2/prefix1/precision_classprefix_dog",
        }
