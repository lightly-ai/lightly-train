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
from typing import Any

from pytorch_lightning.loggers import Logger

from lightly_train._configs import validate
from lightly_train._loggers.jsonl import JSONLLogger
from lightly_train._loggers.logger_args import LoggerArgs
from lightly_train._loggers.mlflow import MLFlowLogger
from lightly_train._loggers.tensorboard import TensorBoardLogger
from lightly_train._loggers.wandb import WandbLogger

logger = logging.getLogger(__name__)


def get_logger_args(loggers: dict[str, Any] | LoggerArgs | None) -> LoggerArgs:
    if isinstance(loggers, LoggerArgs):
        return loggers
    loggers = {} if loggers is None else loggers
    return validate.pydantic_model_validate(LoggerArgs, loggers)


def get_loggers(logger_args: LoggerArgs, out: Path) -> list[Logger]:
    """Get logger instances based on the provided configuration.

    All loggers are configured with the same output directory 'out'.

    Args:
        logger_args:
            Configuration for the loggers.
        out:
            Path to the output directory.

    Returns:
        List of loggers.
    """
    loggers: list[Logger] = []

    if logger_args.jsonl is not None:
        logger.debug(f"Using jsonl logger with args {logger_args.jsonl}")
        loggers.append(JSONLLogger(save_dir=out, **logger_args.jsonl.model_dump()))
    if logger_args.mlflow is not None:
        logger.debug(f"Using mlflow logger with args {logger_args.mlflow}")
        loggers.append(MLFlowLogger(save_dir=out, **logger_args.mlflow.model_dump()))
    if logger_args.tensorboard is not None:
        logger.debug(f"Using tensorboard logger with args {logger_args.tensorboard}")
        loggers.append(
            TensorBoardLogger(save_dir=out, **logger_args.tensorboard.model_dump())
        )
    if logger_args.wandb is not None:
        logger.debug(f"Using wandb logger with args {logger_args.wandb}")
        loggers.append(WandbLogger(save_dir=out, **logger_args.wandb.model_dump()))

    logger.debug(f"Using loggers {[log.__class__.__name__ for log in loggers]}.")
    return loggers
