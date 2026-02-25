#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections.abc import Mapping, Sequence

from torch import Tensor
from torchmetrics import ClasswiseWrapper, Metric, MetricCollection


class ClasswiseMetricCollection(MetricCollection):  # type: ignore[misc]
    """Helper class to compute classwise metrics for a collection of metrics."""

    _SEPARATOR = "<SEP>"

    def __init__(
        self,
        metrics: Mapping[str, Metric] | MetricCollection,
        class_names: Sequence[str],
        prefix: str | None = None,
        postfix: str | None = None,
        classwise_prefix: str | None = None,
        compute_groups: bool | Sequence[Sequence[str]] = True,
    ) -> None:
        wrapper_prefix = self._SEPARATOR
        if classwise_prefix is not None:
            wrapper_prefix = f"{classwise_prefix}_{self._SEPARATOR}"
        wrapped_metrics = {
            name: ClasswiseWrapper(metric, labels=class_names, prefix=wrapper_prefix)  # type: ignore
            for name, metric in metrics.items()
        }
        super().__init__(
            metrics=wrapped_metrics,  # type: ignore
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups,  # type: ignore
        )

    def compute(self) -> dict[str, Tensor]:  # type: ignore[override]
        """Compute metrics and convert into a flat dictionary"""
        metrics = super().compute()
        result: dict[str, Tensor] = {}
        for name, value in metrics.items():
            if isinstance(value, dict):
                # Multiple metrics for each class (e.g. map, map_50, etc.).
                for metric_name, metric_value in value.items():
                    new_name = name.replace(f"{self._SEPARATOR}", metric_name)
                    result[new_name] = metric_value
            else:
                new_name = name.replace(f"{self._SEPARATOR}", "")
                result[new_name] = value
        return result
