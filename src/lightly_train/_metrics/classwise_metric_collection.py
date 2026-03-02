#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections.abc import Mapping, Sequence

from torchmetrics import ClasswiseWrapper, Metric, MetricCollection


class ClasswiseMetricCollection(MetricCollection):  # type: ignore[misc]
    """Helper class to compute classwise metrics for a collection of metrics.

    This class mostly fixes some prefix/suffix issues when using ClasswiseWrapper inside
    a MetricCollection.

    Only use this for metrics that do not return non-classwise and classwise metrics
    at the same time. E.g. MeanAveragePrecision returns both overall mAP and classwise AP,
    so it should not be wrapped in this class.
    """

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

    def compute(self) -> dict[str, float]:  # type: ignore[override]
        """Compute metrics and convert into a flat dictionary"""
        metrics = super().compute()
        result: dict[str, float] = {}
        for name, value in metrics.items():
            new_name = name.replace(f"{self._SEPARATOR}", "")
            result[new_name] = float(value)
        return result
