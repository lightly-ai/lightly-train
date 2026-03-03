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

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleDict

from lightly_train import _torch_helpers
from lightly_train._models import package_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class ImageClassificationMultihead(TaskModel):
    model_suffix = "classification-multihead"

    def __init__(
        self,
        *,
        model: str,
        classes: dict[int, str],
        head_names: list[str],
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """Multi-head image classification model with a frozen backbone.

        Args:
            model:
                A string specifying the model name. It must be in the
                format "{package_name}/{backbone_name}". For example, "dinov3/vitt16" or
                "dinov3/vitt16-classification-multihead".
            classes:
                A dict mapping the class ID to the class name.
            head_names:
                List of head names. One classification head is created for each name.
                Head names should follow the format `head_lr{value}` where value is the
                learning rate formatted without trailing zeros and dots replaced with
                underscores (e.g., "head_lr0_001" for lr=0.001).
            image_size:
                The size of the input images.
            image_normalize:
                A dict containing the mean and standard deviation for normalizing
                the input images. The dict must contain the keys "mean" and "std".
                Example: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}.
                This is used to normalize the input images before passing them to the
                model. If None, no normalization is applied.
            backbone_weights:
                Optional path to a checkpoint file containing the backbone weights. If
                provided, the weights are loaded into the model passed via `model`.
            backbone_args:
                Additional arguments for the backbone. Only used if `model` is a string.
                The arguments are passed to the model when it is instantiated.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        super().__init__(locals(), ignore_args={"backbone_weights", "load_weights"})
        parsed_name = self.parse_model_name(model=model)
        self.model_name = parsed_name["model_name"]
        self.classes = classes
        self.head_names = head_names
        self.image_size = image_size
        self.image_normalize = image_normalize

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

        num_input_channels = (
            3 if self.image_normalize is None else len(self.image_normalize["mean"])
        )

        backbone_model_args = {}
        if model.startswith("dinov2/"):
            backbone_model_args["drop_path_rate"] = 0.0
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)

        self.backbone = package_helpers.get_wrapped_model(
            model=parsed_name["backbone_name"],
            num_input_channels=num_input_channels,
            model_args=backbone_model_args,
            load_weights=load_weights and backbone_weights is None,
        )

        try:
            mask_token = self.backbone.mask_token  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            # The classification model does not use the mask token. We disable grads
            # for the mask token to avoid issues with DDP and find_unused_parameters.
            mask_token.requires_grad = False

        # Load the backbone weights if a path is provided.
        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

        # Backbone is always frozen for multi-head training.
        self.freeze_backbone()

        feature_dim = self.backbone.feature_dim()

        # Create multiple classification heads, one per head name.
        self.class_heads = ModuleDict()
        for head_name in self.head_names:
            head = Linear(feature_dim, len(self.classes))
            head.weight.data.normal_(mean=0.0, std=0.01)
            head.bias.data.zero_()
            self.class_heads[head_name] = head

        _torch_helpers.register_load_state_dict_pre_hook(
            self, class_heads_reuse_or_reinit_hook
        )

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}" for name in package_helpers.list_model_names()
        ]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            package_name, _ = package_helpers.parse_model_name(model=model)
        except ValueError:
            return False
        try:
            package_helpers.get_package(package_name)
        except ValueError:
            return False
        return True

    @classmethod
    def parse_model_name(cls, model: str) -> dict[str, str]:
        model_name = model
        backbone_name = model
        # Suffix is optional as this class supports any backbone model.
        if model.endswith(f"-{cls.model_suffix}"):
            backbone_name = model[: -len(f"-{cls.model_suffix}")]
        else:
            model_name = f"{model}-{cls.model_suffix}"

        return {
            "model_name": model_name,
            "backbone_name": backbone_name,
        }

    def forward_train(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass for training. Returns dict mapping head names to logits."""
        features = self.backbone.forward_pool(self.backbone.forward_features(x))
        pooled_features = features["pooled_features"]  # (B, C, H, W)
        flat_features = pooled_features.flatten(start_dim=1)  # (B, C)

        # Forward through all heads.
        logits = {}
        for head_name, head in self.class_heads.items():
            logits[head_name] = head(flat_features)  # (B, num_classes)

        return logits

    def load_backbone_weights(self, path: PathLike) -> None:
        """Load backbone weights from a checkpoint file.

        Args:
            path: Path to a .pt file, e.g., exported_last.pt.
        """

        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights file not found: '{path}'")

        # Load the checkpoint.
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        unwrapped_backbone = self.backbone.get_model()
        missing, unexpected = unwrapped_backbone.load_state_dict(
            state_dict, strict=False
        )

        # Log missing and unexpected keys.
        if missing or unexpected:
            if missing:
                logger.warning(f"Missing keys when loading backbone: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading backbone: {unexpected}")
        else:
            logger.info(f"Backbone weights loaded from '{path}'")

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("model."):
                name = name[len("model.") :]
                new_state_dict[name] = param
        self.load_state_dict(new_state_dict, strict=True)

    def freeze_backbone(self) -> None:
        self.backbone.eval()  # type: ignore[attr-defined]
        self.backbone.requires_grad_(False)  # type: ignore[attr-defined]


def class_heads_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reuse or reinitialize classification heads when number of classes changes."""
    class_heads_module = getattr(module, "class_heads", None)
    if class_heads_module is None:
        return

    # Check each head in the module.
    for head_name, head_module in class_heads_module.items():
        head_weight_key = f"{prefix}class_heads.{head_name}.weight"
        head_bias_key = f"{prefix}class_heads.{head_name}.bias"
        head_weight = state_dict.get(head_weight_key)

        if head_weight is None:
            continue

        num_classes_state = head_weight.shape[0]
        num_classes_module = head_module.out_features

        if num_classes_state == num_classes_module:
            continue
        else:
            logger.info(
                f"Checkpoint provides {num_classes_state} classes but module expects "
                f"{num_classes_module} for head '{head_name}'. Reinitializing head.",
            )

            # Keep the module initialization by overwriting the checkpoint weights with
            # the current parameter tensors.
            state_dict[head_weight_key] = head_module.weight.detach().clone()
            state_dict[head_bias_key] = head_module.bias.detach().clone()
