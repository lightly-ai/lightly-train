#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
from typing import Any, Literal

from pytorch_lightning.loggers import MLFlowLogger as LightningMLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from typing_extensions import override

from lightly_train._configs.config import PydanticConfig


class MLFlowLoggerArgs(PydanticConfig):
    experiment_name: str = ""
    run_name: str | None = None
    tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
    tags: dict[str, Any] | None = None
    save_dir: str | None = "./mlruns"
    log_model: Literal[True, False, "all"] = False
    prefix: str = ""
    artifact_location: str | None = None
    run_id: str | None = None


#TODO: implement log_image, system info logging
MLFlowLogger = LightningMLFlowLogger

