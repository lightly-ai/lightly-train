#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal

from lightly_train._configs.config import PydanticConfig


class MetricArgs(PydanticConfig):
    """Base class for individual metric arguments.

    Class Attributes:
        watch_mode:
            Whether to maximize ("max") or minimize ("min") this metric
            when used as a watch metric. Defaults to "max" (higher is better).
            Subclasses that represent error metrics (e.g., hamming distance)
            should override with "min".
    """

    watch_mode: ClassVar[Literal["min", "max"]] = "max"

    def supports_classwise(self) -> bool:
        """Whether this metric supports classwise computation.

        Returns False for metrics like topk>1 accuracy that don't make sense classwise.
        """
        raise NotImplementedError


def translate_watch_metric(watch_metric: str, split: str) -> str:
    """Translate a watch_metric key to use the given split prefix.

    Replaces the split part of the watch_metric with the provided split, so that
    TaskMetric instances for different splits (train/val) produce a best_metric_key
    that matches their own computed metric keys.

    Examples:
        "val_metric/miou", "train"  -> "train_metric/miou"
        "val_loss",         "train"  -> "train_loss"
        "val_metric/miou", "val"    -> "val_metric/miou"  (unchanged)

    Args:
        watch_metric: Full metric key (e.g., "val_metric/miou", "val_loss").
        split: Target split name (e.g., "train", "val").

    Returns:
        watch_metric with the original split prefix replaced by split.
    """
    watch_split = watch_metric.split("_")[0]
    return split + watch_metric[len(watch_split) :]


def derive_best_metric_mode(
    watch_metric: str,
    split: str,
    quality_metric_modes: Mapping[str, Literal["min", "max"]],
    loss_names: Sequence[str],
) -> Literal["min", "max"]:
    """Derive best_metric_mode from a watch_metric key.

    Looks up the watch_metric in:
    1. quality_metric_modes (full keys like "{split}_metric/miou")
    2. loss keys ("{split}_loss" or "{split}_loss/{name}") — always "min"

    If watch_metric is not found in the valid keys for the given split,
    returns "max" as a safe default (e.g., when split="train" but
    watch_metric references a "val_*" key).

    Args:
        watch_metric: Full metric key to watch (e.g., "val_metric/miou", "val_loss").
        split: Current split (e.g., "val", "train").
        quality_metric_modes: Mapping from full quality metric key to mode.
            Built by the caller from their MetricArgs fields.
        loss_names: Loss names tracked (e.g., ["loss", "loss_vfl"]).

    Returns:
        "min" or "max" mode for the watch_metric.
    """
    # Check quality metrics
    if watch_metric in quality_metric_modes:
        return quality_metric_modes[watch_metric]

    # Check loss keys (always "min")
    loss_keys: set[str] = set()
    for name in loss_names:
        if name == "loss":
            loss_keys.add(f"{split}_loss")
        else:
            loss_keys.add(f"{split}_loss/{name}")
    if watch_metric in loss_keys:
        return "min"

    # watch_metric not found for this split — safe default
    return "max"
