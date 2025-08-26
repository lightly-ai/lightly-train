#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
try:
    from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
except ImportError:
    SystemMetricsMonitor = None  # type: ignore[misc, assignment]
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
            if isinstance(logger, MLFlowLogger) and SystemMetricsMonitor is not None:
                self.system_monitor = SystemMetricsMonitor(  # type: ignore[no-untyped-call]
                    run_id=logger.run_id,
                )
                self.system_monitor.start()  # type: ignore[no-untyped-call]
                with open(trainer.default_root_dir + "/train.log", "r") as _f:
                    logger.experiment.log_text(
                        run_id=logger.run_id,
                        text=_f.read(),
                        artifact_file="logs/train-start.log",
                    )
                break

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.system_monitor is not None:
            self.system_monitor.finish()  # type: ignore[no-untyped-call]
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                with open(trainer.default_root_dir + "/train.log", "r") as _f:
                    logger.experiment.log_text(
                        run_id=logger.run_id,
                        text=_f.read(),
                        artifact_file="logs/train-end.log",
                    )
                break
