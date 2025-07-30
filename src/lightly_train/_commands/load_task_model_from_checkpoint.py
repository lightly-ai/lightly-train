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

from lightly_train import _logging
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def load_task_model_from_checkpoint(
    *,
    checkpoint: PathLike,
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    """Load a task model from a checkpoint file.

    Args:
        checkpoint:
            Path to the LightlyTrain checkpoint that contains the model.
        device:
            Device to load the model on. If None, the model will be loaded onto a GPU
            (`"cuda"` or `"mps"`) if available, and otherwise fall back to CPU.

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
    task_model = task_model.to(device)
    logger.info(f"Loaded model from {checkpoint_path} on device {task_model.device}.")
    return task_model


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the device to load the model on."""
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    elif device is None:
        if torch.cuda.is_available():
            # Return the default CUDA device if available.
            return torch.device("cuda")
        elif device is None and torch.backends.mps.is_available():
            # Return the default MPS device if available.
            return torch.device("mps")
        else:
            logger.debug("No GPU found, falling back to CPU.")
            return torch.device("cpu")
    else:
        raise ValueError(
            f"Invalid device: {device}. Must be 'cpu', 'cuda', 'mps', a torch.device, or None."
        )


class LoadTaskModelConfig(PydanticConfig):
    checkpoint: PathLike
    device: str | None
