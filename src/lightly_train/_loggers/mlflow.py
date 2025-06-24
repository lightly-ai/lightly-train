#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from lightning_utilities.core.imports import RequirementCache
from PIL.Image import Image
from pydantic import validator
from pytorch_lightning.loggers import MLFlowLogger as LightningMLFlowLogger
from pytorch_lightning.utilities import rank_zero_only

from lightly_train._configs.config import PydanticConfig
from lightly_train._env import Env

logger = logging.getLogger(__name__)

PYTORCH_LIGHTNING_BUG_VERSIONS = ["2.5.1", "2.5.1.post0"]
PL_BUG_VERSION_INSTALLED = any(
    [
        RequirementCache(f"pytorch-lightning=={version}")
        for version in PYTORCH_LIGHTNING_BUG_VERSIONS
    ]
)


class MLFlowLoggerArgs(PydanticConfig):
    experiment_name: str = ""
    run_name: str | None = None
    tracking_uri: str | None = Env.MLFLOW_TRACKING_URI.value
    tags: dict[str, Any] | None = None
    log_model: Literal[True, False, "all"] = False
    prefix: str = ""
    artifact_location: str | None = None
    run_id: str | None = None

    @validator("log_model")  # type: ignore[untyped]
    def validate_log_model(
        cls, v: Literal[True, False, "all"]
    ) -> Literal[True, False, "all"]:
        if v not in [True, False, "all"]:
            raise ValueError("log_model must be one of True, False or 'all'")
        if v in [True, "all"]:
            if PL_BUG_VERSION_INSTALLED:
                logger.warning(
                    f"Due to a bug in pytorch_lightning {PYTORCH_LIGHTNING_BUG_VERSIONS} "
                    "(see https://github.com/Lightning-AI/pytorch-lightning/issues/20822), "
                    "logging models with MLFlowLogger is not possible. If you want to log "
                    "models, please install pytorch-lightning as: pip install "
                    f'"pytorch_lightning>=2.1,!={PYTORCH_LIGHTNING_BUG_VERSIONS}".'
                )
                return False
            return v
        return v


class MLFlowLogger(LightningMLFlowLogger):
    def __init__(
        self,
        experiment_name: str = "lightly_train_logs",
        run_name: str | None = None,
        tracking_uri: str | None = Env.MLFLOW_TRACKING_URI.value,
        tags: dict[str, Any] | None = None,
        save_dir: Path | None = Path("./mlruns"),
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
    ) -> None:
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
