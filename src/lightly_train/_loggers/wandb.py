#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

from pytorch_lightning.loggers import WandbLogger as LightningWandbLogger

from lightly_train._configs.config import PydanticConfig


class WandbLoggerArgs(PydanticConfig):
    name: str | None = None
    version: str | None = None
    offline: bool = False
    anonymous: bool | None = None
    project: str | None = None
    log_model: bool | Literal["all"] = False
    prefix: str = ""
    checkpoint_name: str | None = None


# No customizations necessary.
WandbLogger = LightningWandbLogger
