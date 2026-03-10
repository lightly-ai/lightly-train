#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

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
        """Whether this metric supports classwise computation."""
        raise NotImplementedError

    def get_metric_names(self) -> list[str]:
        """Return bare metric names (no split/prefix) produced by compute().

        Names must be prefixes of the dict keys that would appear in the metric collection
        result after stripping the "{split}_metric/" prefix.

        E.g., for a metric that produces "train_metric/miou" this should return ["miou"].
        """
        raise NotImplementedError
