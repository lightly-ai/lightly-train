# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightly_train._configs.config import PydanticConfig

if TYPE_CHECKING:
    from lightly_train._metrics.base.task_metric import TaskMetric


class TaskMetricArgs(PydanticConfig):
    """Base class for task-specific metrics collection configurations."""

    def get_metrics(self, **extra_args: Any) -> TaskMetric:
        """Create TaskMetric instance with all configured metrics.

        Args:
            **extra_args: Runtime arguments passed to TaskMetric constructor
                         (e.g., num_classes, prefix, class_names, etc.)

        Returns:
            TaskMetric instance for the task.
        """
        raise NotImplementedError
