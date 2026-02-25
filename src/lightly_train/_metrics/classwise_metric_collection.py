#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections.abc import Mapping, Sequence
from typing import Any

from torchmetrics import ClasswiseWrapper, Metric, MetricCollection


class ClasswiseMetricCollection(MetricCollection):  # type: ignore[misc]
    """Renames classwise metric keys to handle class names with underscores.

    Replaces unique separator with underscore, avoiding conflicts when class names
    themselves contain underscores (e.g., "cat__type_a").
    """

    _SEPARATOR = "<SEP>"

    def __init__(
        self,
        metrics: Mapping[str, Metric] | MetricCollection,
        class_names: Sequence[str],
        prefix: str | None = None,
        postfix: str | None = None,
        compute_groups: bool | Sequence[Sequence[str]] = True,
    ) -> None:
        wrapped_metrics = {
            name: ClasswiseWrapper(metric, labels=class_names, prefix=self._SEPARATOR)  # type: ignore
            for name, metric in metrics.items()
        }
        super().__init__(
            metrics=wrapped_metrics,  # type: ignore
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups,  # type: ignore
        )

    def compute(self) -> dict[str, Any]:  # type: ignore[override]
        """Compute metrics and rename keys by replacing separator with underscore."""
        result = super().compute()
        # ClasswiseWrapper joins metric_name with prefix as: metric_name + "_" + prefix + class_name
        # So with prefix="<SEP>" we get: metric_name_<SEP>class_name
        # Replace "_<SEP>" with "_" to get the desired format: metric_name_class_name
        return {
            key.replace(f"_{self._SEPARATOR}", "_"): value
            for key, value in result.items()
        }
