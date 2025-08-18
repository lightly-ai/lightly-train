#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module

from lightly_train.types import PathLike


class TaskModel(Module):
    """Base class for task-specific models that the user interacts with.

    Must implement the forward method for inference. Must be pure PyTorch and not rely
    on Fabric or Lightning modules.
    """

    model_suffix: str

    def __init__(self, init_args: dict[str, Any], ignore_args: set[str] | None = None):
        """
        Args:
            init_args:
                Arguments used to initialize the model. We save those to make it easy
                to serialize and load the model again.
            ignore_args:
                Arguments in init_args that should be ignored. This is useful to ignore
                arguments that are not relevant for serialization, such as
                `backbone_weights` which is not relevant anymore after the model is
                loaded for the first time.
        """
        super().__init__()
        ignore_args = set() if ignore_args is None else ignore_args
        ignore_args.update({"self", "__class__"})
        unknown_keys = ignore_args - init_args.keys()
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in ignore_args: {unknown_keys}. "
                "Please contact the Lightly team if you encounter this error."
            )
        self._init_args = {k: v for k, v in init_args.items() if k not in ignore_args}

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        raise NotImplementedError()

    @property
    def init_args(self) -> dict[str, Any]:
        """Returns the arguments used to initialize the model.

        This is useful for serialization of the model.
        """
        return self._init_args

    @property
    def class_path(self) -> str:
        """Returns the class path of the model.

        This is useful for serialization of the model.
        """
        return f"{self.__module__}.{self.__class__.__name__}"

    def predict(self, image: PathLike | PILImage | Tensor) -> Any:
        """Returns predictions for the given image.

        Args:
            image:
                The input image as a path, PIL image, or tensor. Tensors must have shape
                (C, H, W).
        """
        raise NotImplementedError()

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        raise NotImplementedError()
