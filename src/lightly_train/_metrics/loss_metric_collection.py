#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from torch import Tensor
from torch.nn import Module, ModuleDict


class LossMetricCollection(Module):
    """Tracks a collection of loss metrics, one for each loss name."""

    def __init__(
        self, split: str, loss_names: Sequence[str], train_loss_running_mean_window: int
    ) -> None:
        """Create a LossMetricCollection.

        Args:
            split:
                The data split (e.g., "train", "val") for which the metrics are tracked.
                This is used for naming the output metrics.
            loss_names:
                The names of the losses to track. Each loss name will have a separate
                metric.
            train_loss_running_mean_window:
                The window size for computing the running mean of the training loss.
                This is used to smooth the training loss metric. You want to set this to
                the same value as the gradient_accumulation_steps to get a stable training
                loss metric that reflects the actual optimization steps.
        """
        from torchmetrics import MeanMetric as TorchmetricsMeanMetric
        from torchmetrics import Metric as TorchmetricsMetric

        super().__init__()
        self.split = split

        metric_cls: Callable[[], TorchmetricsMetric]
        if split == "train":
            try:
                from torchmetrics.wrappers import (
                    Running as TorchmetricsRunning,  # type: ignore[attr-defined]
                )

                def metric_cls() -> TorchmetricsMetric:
                    return TorchmetricsRunning(
                        TorchmetricsMeanMetric(),
                        window=train_loss_running_mean_window,
                    )

            except ImportError:
                # Fallback for torchmetrics<1.0, which does not have the Running wrapper
                metric_cls = TorchmetricsMeanMetric
        else:
            metric_cls = TorchmetricsMeanMetric

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
