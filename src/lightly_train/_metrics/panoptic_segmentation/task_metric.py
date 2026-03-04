#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field
from torch import Tensor
from torchmetrics import MetricCollection

from lightly_train import _torch_compile
from lightly_train._metrics.loss_metrics import LossMetrics
from lightly_train._metrics.panoptic_segmentation.panoptic_quality import (
    PanopticQualityArgs,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
    get_watch_metric_mode,
)


class PanopticSegmentationTaskMetricArgs(TaskMetricArgs):
    watch_metric: str = "val_metric/pq"
    classwise: bool = False
    train: bool = False
    pq: PanopticQualityArgs | None = Field(default_factory=PanopticQualityArgs)


class PanopticSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for panoptic segmentation tasks."""

    def __init__(
        self,
        *,
        task_metric_args: PanopticSegmentationTaskMetricArgs,
        split: str,
        things: Sequence[int],
        stuffs: Sequence[int],
        thing_class_names: Sequence[str],
        stuff_class_names: Sequence[str],
        loss_names: Sequence[str],
        init_metrics: bool | None = None,
    ) -> None:
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.things = things
        self.stuffs = stuffs
        self.watch_metric = task_metric_args.watch_metric
        self.watch_metric_mode = get_watch_metric_mode(
            task_metric_args, list(loss_names), task_metric_args.watch_metric
        )

        if init_metrics is None:
            init_metrics = task_metric_args.train if split == "train" else True

        metrics = {}
        if init_metrics and task_metric_args.pq is not None:
            metrics = task_metric_args.pq.get_metrics(
                prefix=f"{split}_metric",
                classwise=False,
                thing_class_names=thing_class_names,
                stuff_class_names=stuff_class_names,
                things=things,
                stuffs=stuffs,
            )
        self.metrics = MetricCollection(metrics)  # type: ignore

        metrics_classwise = {}
        if (
            init_metrics
            and task_metric_args.classwise
            and task_metric_args.pq is not None
        ):
            metrics_classwise = task_metric_args.pq.get_metrics(
                prefix=f"{split}_metric",
                classwise=True,
                thing_class_names=thing_class_names,
                stuff_class_names=stuff_class_names,
                things=things,
                stuffs=stuffs,
            )
        self.metrics_classwise = MetricCollection(metrics_classwise)  # type: ignore
        self.loss_metrics = LossMetrics(split=split, loss_names=loss_names)

    @_torch_compile.disable_compile
    def update(
        self,
        preds: Tensor,
        target: Tensor,
    ) -> None:
        """Update all metrics with inputs.

        Args:
            preds: Prediction tensor of shape (B, H, W, 2) where last dim is (class_id, instance_id)
            target: Target tensor of shape (B, H, W, 2) where last dim is (class_id, instance_id)
        """
        self.metrics.update(preds, target)
        self.metrics_classwise.update(preds, target)

    @_torch_compile.disable_compile
    def update_loss(self, loss_dict: Mapping[str, Tensor], weight: int) -> None:
        self.loss_metrics.update(loss_dict=loss_dict, weight=weight)

    def compute(self) -> MetricComputeResult:
        result = self.loss_metrics.compute()
        result.update(self.metrics.compute())
        result.update(self.metrics_classwise.compute())
        result = {name: float(value) for name, value in result.items()}
        best_value = result.get(self.watch_metric)
        return MetricComputeResult(
            metrics=result,
            watch_metric=self.watch_metric if best_value is not None else None,
            watch_metric_value=float(best_value) if best_value is not None else None,
            watch_metric_mode=(
                self.watch_metric_mode if best_value is not None else None
            ),
            best_head_name=None,
            best_head_metrics=None,
        )
