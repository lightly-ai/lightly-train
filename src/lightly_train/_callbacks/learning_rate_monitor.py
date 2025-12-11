#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pytorch_lightning.callbacks import (
    LearningRateMonitor as LightningLearningRateMonitor,
)


class LearningRateMonitor(LightningLearningRateMonitor):
    def _get_optimizer_stats(self, *args, **kwargs) -> dict[str, float]:  # type: ignore[no-untyped-def]
        # This fixes https://github.com/Lightning-AI/pytorch-lightning/issues/20250
        # The proper fix would be to add the float conversion in LightlySSL here:
        # https://github.com/lightly-ai/lightly/blob/ee30cd481d68862c80de4ef45920cfe1ab1f67b1/lightly/utils/scheduler.py#L67
        # But LightlyTrain has to be backwards compatible with older Lightly versions
        # so we add the fix here for now.
        stats = super()._get_optimizer_stats(*args, **kwargs)
        stats = {name: float(value) for name, value in stats.items()}
        return stats
