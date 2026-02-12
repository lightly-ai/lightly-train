#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any

from torch.nn import Module

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
   - Example: ClassificationTaskMetric(metric_args, num_classes=10, prefix="val_metric/")
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

    def __init__(self) -> None:
        super().__init__()

    def items(self) -> dict[str, Any]:
        """Get all metric instances for adding to log_dict.

        Returns:
            Dictionary of metric name -> metric instance.
            These are the actual metric objects that will be updated during training.
        """
        raise NotImplementedError

    def get_display_names(self) -> dict[str, str]:
        """Get display names for metrics (for logging).

        Returns:
            Dictionary mapping metric names to human-readable display names.
            Example: {"val_metric/top1_acc_micro": "Val Top-1 Acc (Micro)"}
        """
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        """Compute all metrics and return float values for logging.

        Returns float values that can be directly logged with fabric.log().

        Returns:
            Dictionary mapping metric names to computed float values.
        """
        # Default implementation - just compute all metrics
        result = {}
        for name, metric in self.items().items():
            from torchmetrics import Metric

            if isinstance(metric, Metric):
                value = metric.compute()
                if hasattr(value, "item"):
                    result[name] = float(value.item())
                else:
                    result[name] = float(value)
        return result

    def reset(self) -> None:
        """Reset all metrics.

        This is called by reset_metrics() helper.
        """
        for metric in self.items().values():
            from torchmetrics import Metric

            if isinstance(metric, Metric):
                metric.reset()
