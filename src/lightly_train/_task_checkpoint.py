#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Literal

from pydantic import model_validator
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig

logger = logging.getLogger(__name__)


class TaskSaveCheckpointArgs(PydanticConfig):
    save_every_num_steps: int = 1000
    save_last: bool = True
    save_best: bool = True
    watch_metric: str
    mode: Literal["min", "max"]

    @model_validator(mode="after")
    def _warn_if_no_checkpoints(self) -> Self:
        if not self.save_last and not self.save_best:
            logger.warning(
                "No checkpoints will be saved because both 'save_last' and 'save_best' "
                "are disabled. At least one of them should be enabled if checkpoint "
                "artifacts are required."
            )
        return self
