#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import importlib
from typing import Literal

import torch

from lightly_train._commands import common_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike


def load_model_from_checkpoint(
    checkpoint: PathLike,
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    """Load a model from a checkpoint file.

    Args:
        checkpoint:
            Path to the checkpoint file. For example "out/checkpoints/last.ckpt".
        device:
            Device to load the model on. If None, the model will be loaded onto a GPU
            (`"cuda"` or `"mps"`) if available, and otherwise fall back to CPU.

    Returns:
        The loaded model.
    """
    device = _resolve_device(device)
    checkpoint = common_helpers.get_checkpoint_path(checkpoint=checkpoint)
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)

    # Import the model class dynamically
    module_path, class_name = ckpt["model_class_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Create model instance
    model: TaskModel = model_class(**ckpt["model_init_args"])
    model.load_train_state_dict(state_dict=ckpt["train_model"])
    model.eval()

    model = model.to(device)
    return model


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
            return torch.device("cpu")
    else:
        raise ValueError(
            f"Invalid device: {device}. Must be 'cpu', 'cuda', 'mps', a torch.device, or None."
        )
