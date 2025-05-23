#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from PIL.Image import Image
from pytorch_lightning.loggers import MLFlowLogger as LightningMLFlowLogger
from pytorch_lightning.utilities import rank_zero_only

from lightly_train._configs.config import PydanticConfig


class MLFlowLoggerArgs(PydanticConfig):
    experiment_name: str = ""
    run_name: str | None = None
    tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI")
    tags: dict[str, Any] | None = None
    log_model: Literal[True, False, "all"] = False
    prefix: str = ""
    artifact_location: str | None = None
    run_id: str | None = None


class MLFlowLogger(LightningMLFlowLogger):
    def __init__(
        self,
        experiment_name: str = "lightly_train_logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        tags: dict[str, Any] | None = None,
        save_dir: Path | None = Path("./mlruns"),
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
    ) -> None:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=str(save_dir),
            log_model=log_model,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
        )
        self.save_temp_dir = str(save_dir)

    @rank_zero_only  # type: ignore[misc]
    def log_image(self, key: str, images: list[Image], step: int | None = None) -> None:
        for image in images:
            self.experiment.log_image(
                run_id=self.run_id,
                image=image,
                key=key,
                step=step,
            )
