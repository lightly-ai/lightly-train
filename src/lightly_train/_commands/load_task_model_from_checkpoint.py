#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import distributed

from lightly_train._task_models.task_model import TaskModel
import lightly_train._distributed as train_distributed
from lightly_train import _logging
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models import task_model_helpers
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

def load_task_model_from_checkpoint(
        *,
        checkpoint: PathLike,
        device: Literal["cpu", "cuda", "mps"] | None = None,
) -> TaskModel:
    """Load a task model from a checkpoint file.

    Args:
        checkpoint:
            Path to the LightlyTrain checkpoint that contains the model.
        device:
            Device to load the model on. If None, the model will be loaded to the device
            it was saved on.

    Returns:
        The loaded model.
    """
    config = LoadTaskModelConfig(**locals())
    return load_task_model_from_config(config=config)

def load_task_model_from_config(config: LoadTaskModelConfig) -> TaskModel:
    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    checkpoint_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)
    task_model = task_model_helpers.load_task_model_from_checkpoint(
        checkpoint=checkpoint_path
    )
    task_model.eval()

    device = _resolve_device(config.device)
    if device is not None:
        task_model.to(device)
    logger.info(f"Loaded model from {checkpoint_path} on device {task_model.device}.")
    return task_model


def _resolve_device(device: str | None) -> torch.device | None:
    """Resolve the device to load the model on."""
    if device is None:
        return None
    elif device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        return torch.device(train_distributed.get_local_rank() or "cuda")
    elif device == "mps":
        return torch.device("mps")
    else:
        raise TypeError(f"Expected str or None, got {type(device)}")


class LoadTaskModelConfig(PydanticConfig):
    checkpoint: PathLike
    device: str | None