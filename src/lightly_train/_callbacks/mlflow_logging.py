#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from lightly_train._configs.config import PydanticConfig
from lightly_train._loggers.mlflow import MLFlowLogger


class MLFlowLoggingArgs(PydanticConfig):
    pass


class MLFlowLogging(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.system_monitor = None
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                self.system_monitor = SystemMetricsMonitor(
                    run_id=logger.run_id,
                )
                self.system_monitor.start()
                logger.experiment.log_artifact(
                    run_id=logger.run_id,
                    local_path=trainer.default_root_dir + "/train.log",
                    artifact_path="logs",
                )
                break

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.system_monitor is not None:
            self.system_monitor.finish()
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_artifact(
                    run_id=logger.run_id,
                    local_path=trainer.default_root_dir + "/train.log",
                    artifact_path="logs",
                )
                break
