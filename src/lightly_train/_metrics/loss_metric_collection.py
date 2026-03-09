#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial

from torch import Tensor
from torch.nn import Module, ModuleDict


class LossMetricCollection(Module):
    """Tracks a collection of loss metrics, one for each loss name."""

    def __init__(
        self, split: str, loss_names: Sequence[str], running_mean_window: int = 20
    ) -> None:
        from torchmetrics import MeanMetric as TorchmetricsMeanMetric

        try:
            # Type ignore for old torchmetrics versions
            from torchmetrics import (  # type: ignore[attr-defined]
                RunningMean as TorchmetricsRunningMean,
            )

            running_mean_cls = partial(
                TorchmetricsRunningMean, window=running_mean_window
            )
        except ImportError:
            # Fall back to MeanMetric for old torchmetrics versions
            running_mean_cls = TorchmetricsMeanMetric  # type: ignore

        super().__init__()
        self.split = split
        # For training we only track the losses of the most recent batches to keep an
        # up-to-date estimate of the current loss. We do not only track the last loss
        # value because of gradient accumulation which can cause the loss to fluctuate
        # a lot from accumulation step to accumulation step.
        metric_cls = running_mean_cls if split == "train" else TorchmetricsMeanMetric
        self.metrics = ModuleDict({loss_name: metric_cls() for loss_name in loss_names})

    def update(self, loss_dict: Mapping[str, Tensor], weight: int) -> None:
        if loss_dict.keys() != self.metrics.keys():
            raise ValueError(
                f"Loss dict keys {loss_dict.keys()} do not match expected loss names "
                f"{self.metrics.keys()}"
            )
        for loss_name, loss_value in loss_dict.items():
            self.metrics[loss_name].update(loss_value, weight=weight)  # type: ignore

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for loss_name, metric in self.metrics.items():
            value = metric.compute().item()  # type: ignore
            if loss_name == "loss":
                result[f"{self.split}_loss"] = value
            else:
                result[f"{self.split}_loss/{loss_name}"] = value
        return result
