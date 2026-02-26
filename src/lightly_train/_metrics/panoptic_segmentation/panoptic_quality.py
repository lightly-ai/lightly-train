#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection import PanopticQuality as TorchMetricsPanopticQuality

from lightly_train._metrics.metric_args import MetricArgs


class PanopticQualityArgs(MetricArgs):
    """Panoptic Quality metric for panoptic segmentation."""

    def get_metrics(  # type: ignore[override]
        self,
        *,
        prefix: str,
        classwise: bool,
        thing_class_names: Sequence[str],
        stuff_class_names: Sequence[str],
        things: Sequence[int],
        stuffs: Sequence[int],
    ) -> dict[str, Metric]:
        """Create PanopticQuality metric instance for panoptic segmentation.

        Args:
            classwise: If True, compute per-class metrics
            things: List of thing class IDs
            stuffs: List of stuff class IDs

        Returns:
            Dictionary with single "pq" key containing the metric instance
        """
        metrics: dict[str, Metric] = {}

        metrics["pq"] = PanopticQuality(
            prefix=prefix,
            thing_class_names=thing_class_names,
            stuff_class_names=stuff_class_names,
            things=things,
            stuffs=stuffs,
            return_sq_and_rq=True,
            return_per_class=classwise,
        )

        return metrics

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["pq", "sq", "rq"]


class PanopticQuality(TorchMetricsPanopticQuality):
    """Wrapper around torchmetrics PanopticQuality to handle ignored classes."""

    def __init__(
        self,
        *,
        prefix: str,
        things: Sequence[int],
        stuffs: Sequence[int],
        thing_class_names: Sequence[str],
        stuff_class_names: Sequence[str],
        return_sq_and_rq: bool = False,
        return_per_class: bool = False,
    ) -> None:
        super().__init__(
            things=things,
            stuffs=stuffs,
            return_sq_and_rq=return_sq_and_rq,
            return_per_class=return_per_class,
        )
        self.prefix = prefix
        self.thing_class_names = thing_class_names
        self.stuff_class_names = stuff_class_names
        self.class_id_to_name = {
            class_id: name
            for class_id, name in zip(
                list(things) + list(stuffs),
                list(thing_class_names) + list(stuff_class_names),
            )
        }

    def compute(self) -> dict[str, Tensor]:  # type: ignore
        metrics = super().compute()
        result: dict[str, Tensor] = {}
        if self.return_per_class:
            if self.return_sq_and_rq:
                # Metrics has shape (num_classes, 3)
                for class_id in range(len(metrics)):
                    class_name = self.class_id_to_name[class_id]
                    result[f"{self.prefix}_classwise/pq_{class_name}"] = metrics[
                        class_id, 0
                    ]
                    result[f"{self.prefix}_classwise/sq_{class_name}"] = metrics[
                        class_id, 1
                    ]
                    result[f"{self.prefix}_classwise/rq_{class_name}"] = metrics[
                        class_id, 2
                    ]
            else:
                # Metrics has shape (num_classes,)
                for class_id in range(len(metrics)):
                    class_name = self.class_id_to_name[class_id]
                    result[f"{self.prefix}_classwise/pq_{class_name}"] = metrics[
                        class_id
                    ]
        elif self.return_sq_and_rq:
            # Metrics has shape (3,)
            result[f"{self.prefix}/pq"] = metrics[0]
            result[f"{self.prefix}/sq"] = metrics[1]
            result[f"{self.prefix}/rq"] = metrics[2]
        else:
            # Metrics is a single scalar
            result[f"{self.prefix}/pq"] = metrics
        return result
