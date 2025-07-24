#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import importlib

import torch

from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike


def load_task_model_from_checkpoint(checkpoint: PathLike) -> TaskModel:
    """Load a task model from a checkpoint file.

    Args:
        checkpoint:
            Path to the checkpoint file. For example "out/checkpoints/last.ckpt".
    Returns:
        The loaded model.
    """
    ckpt = torch.load(checkpoint, weights_only=False)

    # Import the model class dynamically
    module_path, class_name = ckpt["model_class_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Create model instance
    model: TaskModel = model_class(**ckpt["model_init_args"])
    model.load_train_state_dict(state_dict=ckpt["train_model"])
    return model
