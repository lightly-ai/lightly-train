#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch.nn import Module
from torchmetrics import Metric, MetricCollection

from lightly_train._configs.config import PydanticConfig
from lightly_train._metrics.classwise_metric_collection import (
    ClasswiseMetricCollection,
)
from lightly_train._metrics.metric_args import MetricArgs

"""Base classes for task-specific metrics.

## Structure Overview

The metrics system has three layers:

1. **MetricArgs** (Pydantic): Configuration for individual metrics
   - Example: MulticlassAccuracyArgs(topk=[1, 5], average=["micro", "macro"])
   - Validates configuration at parse time
   - Has .get_metrics() method that creates torchmetrics instances

2. **TaskMetricArgs** (Pydantic): Configuration for all metrics in a task
   - Example: MulticlassClassificationTaskMetricArgs(
        accuracy=..., f1=..., precision=...
    )
   - Groups related MetricArgs for a task
   - Has .get_metrics() method that creates a TaskMetric instance

3. **TaskMetric** (nn.Module): Runtime manager for metrics
   - Example: ClassificationTaskMetric(metric_args, class_names=[...], prefix="val_metric/")
   - Stores actual torchmetrics instances (MetricCollection, MeanMetric)
   - Provides .items() for logging, .get_display_names() for human-readable names
   - Inherits from nn.Module for automatic device handling

## Adding a New Metric

### Step 1: Create MetricArgs class

```python
# In classification/multiclass_metric_args.py
class MulticlassNewMetricArgs(MetricArgs):
    param1: int = 10
    param2: str = "default"

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics = {}
        key = "new_metric"
        metrics[key] = MulticlassNewMetric(
            num_classes=num_classes,
            param1=self.param1,
            param2=self.param2,
            average="none" if classwise else "micro",
        )
        return metrics

    def supports_classwise(self) -> bool:
        return True  # or False if classwise doesn't make sense
```

### Step 2: Add to TaskMetricArgs

```python
# In classification/task_metric_args.py
class MulticlassClassificationTaskMetricArgs(TaskMetricArgs):
    accuracy: MulticlassAccuracyArgs | None = None
    new_metric: MulticlassNewMetricArgs | None = None  # Add here
    # ... other metrics
```

### Step 3: Use in training

```python
# In config or code
metrics_args = MulticlassClassificationTaskMetricArgs(
    accuracy=MulticlassAccuracyArgs(topk=[1, 5]),
    new_metric=MulticlassNewMetricArgs(param1=20),
)

# At runtime
val_metrics = metrics_args.get_metrics(
    num_classes=10,
    prefix="val_metric/",
)
val_metrics.metrics.update(logits, targets)  # Use like any metric
```

That's it! The new metric is now integrated.
"""


@dataclass
class MetricComputeResult:
    """Result of computing all metrics.

    Attributes:
        metrics:
            Dictionary mapping metric names to computed float values.
            For multihead metrics, contains per-head keys (e.g.,
            "val_metric_head/miou_lr0_001") and best-head top-level keys
            (e.g., "val_metric/miou").
        best_metric_key:
            The key of the metric used for model selection
            (e.g., "val_metric/miou"). Used by the training loop for
            checkpointing. None if the watch metric is not found in the
            computed metrics (e.g., a train metric watching a val key).
        best_metric_value:
            The value of the best_metric_key metric.
            None if best_metric_key is None.
        best_head_name:
            Name of the best head (e.g., "lr0_01").
            None for single-head metrics.
        best_head_metrics:
            Metrics of the best head with top-level keys
            (no head suffix). None for single-head metrics.
    """

    metrics: dict[str, float]
    best_metric_key: str | None
    best_metric_value: float | None
    best_head_name: str | None
    best_head_metrics: dict[str, float] | None


class TaskMetricArgs(PydanticConfig):
    """Base class for task-specific metrics collection configurations."""

    # Metric key to watch for best model selection. E.g. "val_metric/f1_macro".
    watch_metric: str

    def iter_metric_args(self, *, classwise: bool) -> dict[str, MetricArgs]:
        """Iterate over all MetricArgs fields in this TaskMetricArgs.

        Returns:
            Dictionary mapping field names to MetricArgs instances.
            Example: {"accuracy": MulticlassAccuracyArgs(...), "f1": MulticlassF1Args(...)}
        """
        metric_args_dict = {}
        for field_name in self.__class__.model_fields:
            individual_metric_args = getattr(self, field_name)
            if not isinstance(individual_metric_args, MetricArgs):
                continue
            if classwise and not individual_metric_args.supports_classwise():
                continue
            metric_args_dict[field_name] = individual_metric_args
        return metric_args_dict

    def build_metric_collection(
        self, *, prefix: str, **kwargs: Any
    ) -> MetricCollection:
        """Build a flat dictionary of metric instances from TaskMetricArgs."""
        all_metrics: dict[str, Metric] = {}
        for metric_arg in self.iter_metric_args(classwise=False).values():
            all_metrics.update(metric_arg.get_metrics(classwise=False, **kwargs))
        return MetricCollection(all_metrics, prefix=prefix)  # type: ignore[arg-type]

    def build_classwise_metric_collection(
        self,
        *,
        log_classwise: bool,
        prefix: str,
        classwise_metrics_args: TaskMetricArgs | None,
        class_names: Sequence[str],
        **kwargs: Any,
    ) -> MetricCollection | None:
        """Build a classwise MetricCollection if log_classwise is True."""
        if not log_classwise:
            return None
        if classwise_metrics_args is None:
            classwise_metrics_args = self.model_copy()
        metrics = self.build_metric_collection(prefix="", **kwargs)
        return ClasswiseMetricCollection(
            metrics=metrics,
            class_names=class_names,
            prefix=prefix,
        )


class TaskMetric(Module):
    """Base class for task-specific metrics container.

    This is a CONTAINER, not a metric itself. It organizes and provides access
    to the actual torchmetrics instances.

    Device Handling:
    - All metrics stored as attributes are automatically detected as child modules
    - Transferred to correct device when .to(device) is called
    - Handled by Lightning Fabric's setup() method

    Responsibilities:
    - Store metric instances (MetricCollections, MeanMetric)
    - Provide access via .items() for logging
    - Provide display names via .get_display_names()
    - Handle task-specific formatting in .compute() if needed
    """

    def __init__(self, task_metric_args: TaskMetricArgs) -> None:
        super().__init__()
        self.task_metric_args = task_metric_args

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return values for logging.

        Returns:
            MetricComputeResult containing metrics dict, best_metric_key, and best_metric_value.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset all metrics."""
        for module in self.modules():
            if isinstance(module, Metric):
                module.reset()
