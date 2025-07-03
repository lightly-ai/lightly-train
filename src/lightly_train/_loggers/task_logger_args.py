#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

from lightly_train._configs.config import PydanticConfig


class TaskLoggerArgs(PydanticConfig):
    log_every_num_steps: int | Literal["auto"] = "auto"

    def resolve_auto(self, steps: int) -> None:
        if self.log_every_num_steps == "auto":
            self.log_every_num_steps = min(100, max(1, steps // 10))
