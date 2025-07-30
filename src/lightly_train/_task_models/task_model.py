#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import torch
from torch.nn import Module


class TaskModel(Module):
    """Base class for task-specific models that the user interacts with.

    Must implement the forward method for inference. Must be pure PyTorch and not rely
    on Fabric or Lightning modules.
    """

    def __init__(self, init_args: dict[str, Any]):
        init_args = init_args.copy()
        init_args.pop("self", None)
        init_args.pop("__class__", None)
        self._init_args = init_args
        super().__init__()

    @property
    def init_args(self) -> dict[str, Any]:
        """Returns the arguments used to initialize the model

        This is useful for serialization of the model.
        """
        return self._init_args

    @property
    def class_path(self) -> str:
        """Returns the class path of the model."""
        return f"{self.__module__}.{self.__class__.__name__}"

    @property
    def device(self) -> torch.device:
        """Returns the device the model is currently on."""
        return next(self.parameters()).device

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        raise NotImplementedError()
