#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import logging
import os

from lightning_fabric.loggers.logger import rank_zero_experiment
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter as CSVExperimentWriter

from lightly_train._configs.config import PydanticConfig
from lightly_train.types import PathLike

log = logging.getLogger(__name__)


# Implementation based on ExperimentWriter for CSVLogger from PyTorch Lightning.
class ExperimentWriter(CSVExperimentWriter):
    r"""Experiment writer for JSONLLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_METRICS_FILE = "metrics.jsonl"

    def save(self) -> None:
        """Save recorded metrics into files."""
        # TODO: Save hparams to a file.
        if not self.metrics:
            return

        file_exists = self._fs.isfile(self.metrics_file_path)

        with self._fs.open(
            self.metrics_file_path, mode=("a" if file_exists else "w"), newline=""
        ) as file:
            file.writelines(f"{json.dumps(metric)}\n" for metric in self.metrics)

        self.metrics: list[dict[str, float]] = []  # reset

    def _check_log_dir_exists(self) -> None:
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty. "
                "Previous log files in this directory can be modified when the new "
                "ones are saved!"
            )


class JSONLLoggerArgs(PydanticConfig):
    flush_logs_every_n_steps: int = 100


class JSONLLogger(CSVLogger):
    """Log to local file system in JSON Lines format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.
    Logs can be loaded with `pandas.read_json("metrics.jsonl", lines=True)`.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from lightly_train.loggers import JSONLLogger
        >>> logger = JSONLLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name.
        version:
            Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next
            available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps:
            How often to flush logs to disk (defaults to every 100 steps).

    """

    def __init__(
        self,
        save_dir: PathLike,
        name: str = "",
        version: int | str | None = "",
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        self._experiment: ExperimentWriter | None  # type: ignore[assignment]

    @property
    @rank_zero_experiment  # type: ignore[misc]
    def experiment(self) -> "ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in
        your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment
