#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Linear, Module, ModuleDict
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train import _torch_helpers
from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SemanticSegmentationMultihead(TaskModel):
    model_suffix = "semantic-segmentation-linear"

    def __init__(
        self,
        *,
        model: str,
        classes: dict[int, str],
        head_names: list[str],
        class_ignore_index: int | None = None,
        image_size: tuple[int, int] = (518, 518),
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """Multi-head semantic segmentation model with a frozen backbone.

        Args:
            model:
                A string specifying the model name. It must be in the
                format "{package_name}/{backbone_name}". For example, "dinov2/vits14-linear".
            classes:
                A dict mapping the class ID to the class name.
            head_names:
                List of head names. One segmentation head is created for each name.
                Head names should follow the format `head_lr{value}` where value is the
                learning rate formatted without trailing zeros and dots replaced with
                underscores (e.g., "head_lr0_001" for lr=0.001).
            class_ignore_index:
                The class index to ignore during training. If provided, an additional
                class output is created for this ignored class.
            image_size:
                The size of the input images as (height, width). Default (518, 518).
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
        self.class_ignore_index = class_ignore_index
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
            # The segmentation model does not use the mask token. We disable grads
            # for the mask token to avoid issues with DDP and find_unused_parameters.
            mask_token.requires_grad = False

        # Load the backbone weights if a path is provided.
        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

        # Backbone is always frozen for multi-head training.
        self.freeze_backbone()

        # Get patch size from backbone if available.
        patch_size: int | None = getattr(self.backbone, "patch_size", None)  # type: ignore[attr-defined]
        self.patch_size = patch_size

        feature_dim = self.backbone.feature_dim()

        # Number of output classes includes the ignore class if specified.
        num_classes = len(self.classes)
        if self.class_ignore_index is not None:
            num_classes += 1

        # Create multiple segmentation heads, one per head name.
        self.class_heads = ModuleDict()
        for head_name in self.head_names:
            head = Linear(feature_dim, num_classes)
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

    @staticmethod
    def _format_head_name(lr: float) -> str:
        """Format learning rate into a head name.

        Args:
            lr: Learning rate value.

        Returns:
            Formatted head name, e.g., "lr0_001" for lr=0.001.
        """
        # Convert to string and replace dot with underscore.
        lr_str = f"{lr:.10f}".rstrip("0").rstrip(".")
        lr_str = lr_str.replace(".", "_")
        return f"lr{lr_str}"

    def forward_train(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass for training. Returns dict mapping head names to logits.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Dict mapping head names to logits tensors of shape (B, K, H, W) where
            K is the number of classes.
        """
        B, _, H, W = x.shape

        # Get the features from the backbone.
        # Features shape: (B, D, H_feat, W_feat) where H_feat and W_feat depend on
        # the backbone architecture.
        features = self.backbone.forward_features(x)["features"]

        # Get feature dimensions for reshaping.
        _, D, H_feat, W_feat = features.shape

        # Reshape features to (B, H*W, D) for head processing.
        flat_features = features.view(B, D, -1).permute(0, 2, 1)  # (B, H*W, D)

        # Forward through all heads.
        logits_dict = {}
        for head_name, head in self.class_heads.items():
            # Apply head: (B, H*W, D) -> (B, H*W, K)
            head_logits = head(flat_features)  # (B, H*W, K)

            # Reshape back to (B, K, H_feat, W_feat).
            head_logits = head_logits.permute(0, 2, 1).reshape(B, -1, H_feat, W_feat)

            # Upsample to match original image resolution if needed.
            if (H_feat, W_feat) != (H, W):
                head_logits = F.interpolate(
                    head_logits, size=(H, W), mode="bilinear", align_corners=False
                )

            logits_dict[head_name] = head_logits

        return logits_dict

    def load_backbone_weights(self, path: PathLike) -> None:
        """Load backbone weights from a checkpoint file.

        Args:
            path: Path to a .pt file, e.g., exported_last.pt.
        """
        import os

        # Check if the file exists.
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}")
            return

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
        """Freeze all backbone parameters."""
        self.backbone.eval()  # type: ignore[attr-defined]
        for param in self.backbone.parameters():
            param.requires_grad = False

    def tile(
        self, images: list[Tensor] | Tensor
    ) -> tuple[list[Tensor], list[tuple[int, int, int, bool]]]:
        """Split images into tiles for handling variable-sized inputs.

        Args:
            images: List of image tensors or a single batched tensor.

        Returns:
            Tuple of (crops, origins) where crops is a list of tiled images and
            origins contains metadata for untiling.
        """
        crops, origins = [], []

        for i, image in enumerate(images):
            h, w = image.shape[-2:]
            long_side_size = max(h, w)
            short_side_size = min(h, w)

            # Is the image tall or wide?
            is_tall = h > w

            # By construction the short side size is equal to the crop size.
            crop_size = short_side_size
            num_crops = math.ceil(long_side_size / crop_size)
            overlap = num_crops * crop_size - long_side_size
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (crop_size - overlap_per_crop))
                end = start + crop_size

                # Image is tall.
                if is_tall:
                    crop = image[:, start:end, :]

                # Image is wide.
                else:
                    crop = image[:, :, start:end]

                # Store the crop.
                crops.append(crop)

                # Store the position of the crop.
                origins.append((i, start, end, is_tall))

        return crops, origins

    def untile(
        self,
        crop_logits: Tensor,
        origins: list[tuple[int, int, int, bool]],
        image_sizes: list[tuple[int, int]],
    ) -> list[Tensor]:
        """Reconstruct full-size logits from tiled predictions.

        Args:
            crop_logits: Tensor of shape (N_crops, K, H, W) containing logits for
                all crops.
            origins: List of tuples containing tile metadata from the tile() method.
            image_sizes: List of original image sizes as (height, width) tuples.

        Returns:
            List of tensors, one per image, with shape (K, H, W).
        """
        logit_sums, logit_counts = [], []

        # Initialize the tensors containing the final predictions.
        for size in image_sizes:
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], *size), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], *size), device=crop_logits.device)
            )

        for crop_index, (image_index, start, end, is_tall) in enumerate(origins):
            # Image is tall.
            if is_tall:
                logit_sums[image_index][:, start:end, :] += crop_logits[crop_index]
                logit_counts[image_index][:, start:end, :] += 1
            # Image is wide.
            else:
                logit_sums[image_index][:, :, start:end] += crop_logits[crop_index]
                logit_counts[image_index][:, :, start:end] += 1

        # Average the logits in the regions of overlap.
        return [
            logit_sum / logit_count
            for logit_sum, logit_count in zip(logit_sums, logit_counts)
        ]

    def _forward_logits(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass with tiling support for inference.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Dict mapping head names to logits tensors of shape (B, K, H, W).
        """
        # Tiling.
        image_sizes = [img.shape[-2:] for img in x]
        crops_list, origins = self.tile(images=x)
        crops = torch.stack(crops_list)
        crop_h, crop_w = crops.shape[-2:]

        # Forward pass through all heads.
        crop_logits_dict = self.forward_train(crops)

        # Interpolate and untile for each head.
        logits_dict = {}
        for head_name, crop_logits in crop_logits_dict.items():
            # Interpolate back to crop size.
            crop_logits = F.interpolate(crop_logits, (crop_h, crop_w), mode="bilinear")

            # Untile to get full image logits.
            logits_list = self.untile(
                crop_logits=crop_logits,
                origins=origins,
                image_sizes=[tuple(size) for size in image_sizes],  # type: ignore[arg-type]
            )
            logits = torch.stack(logits_list)  # (B, K, H, W)
            logits_dict[head_name] = logits

        return logits_dict

    def predict(self, image: PathLike | PILImage | Tensor) -> dict[str, Tensor]:
        """Predict class indices for each pixel in the image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape (C, H, W).

        Returns:
            Dict mapping head names to predicted class index tensors of shape (H, W).
            The values are the actual class IDs from the dataset.
        """
        self._track_inference()
        if self.training:
            self.eval()

        first_param = next(self.parameters())
        device = first_param.device
        dtype = first_param.dtype

        # Load image.
        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        if self.image_normalize is not None:
            x = transforms_functional.normalize(
                x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
            )

        # Crop size is the short side of the training image size. We resize the image
        # such that the short side of the image matches the crop size.
        crop_size = min(self.image_size)
        # (C, H, W) -> (C, H', W')
        x = transforms_functional.resize(x, size=[crop_size])
        x = x.unsqueeze(0)  # (1, C, H', W')

        # Get logits for all heads.
        logits_dict = self._forward_logits(x)

        # Convert logits to predictions for each head.
        predictions = {}
        for head_name, logits in logits_dict.items():
            # (1, K|K+1, H', W')
            if self.class_ignore_index is not None:
                # Restrict logits to known classes only (exclude ignore class).
                logits = logits[:, :-1]  # (1, K, H', W')

            # Interpolate back to original image size.
            logits = F.interpolate(logits, size=(image_h, image_w), mode="bilinear")

            # Get predicted class indices.
            masks = logits.argmax(dim=1)  # (1, H, W)

            # Map internal class indices to actual class IDs.
            masks = self.internal_class_to_class[masks]  # (1, H, W)

            predictions[head_name] = masks[0]  # (H, W)

        return predictions


def class_heads_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reuse or reinitialize segmentation heads when number of classes changes.

    This hook is called before loading a state dict into the model. It checks if the
    number of classes in the checkpoint matches the number of classes expected by the
    model. If they don't match, the head is reinitialized instead of loading the
    checkpoint weights.

    Args:
        module: The module being loaded.
        state_dict: The state dict being loaded.
        prefix: The prefix for keys in the state dict.
        *args: Additional arguments (unused).
        **kwargs: Additional keyword arguments (unused).
    """
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
