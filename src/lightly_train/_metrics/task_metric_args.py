#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import TYPE_CHECKING

from lightly_train._configs.config import PydanticConfig

if TYPE_CHECKING:
    from lightly_train._metrics.task_metric import TaskMetric


class TaskMetricArgs(PydanticConfig):
    """Base class for task-specific metrics collection configurations."""

    def get_metrics(self) -> TaskMetric:
        """Create TaskMetric instance with all configured metrics.

        Subclasses must implement this with their specific runtime arguments.

        Returns:
            TaskMetric instance for the task.
        """
        raise NotImplementedError
