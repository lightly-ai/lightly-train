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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Linear
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class DINOv2LinearSemanticSegmentation(TaskModel):
    model_suffix = "linear"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        class_ignore_index: int | None,
        backbone_freeze: bool,
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]],
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """
        Args:
            model_name:
                The model name. For example "dinov2/vits14-linear".
            classes:
                A dict mapping the class ID to the class name. The dict must only
                contain the classes that the model should predict. It must NOT contain
                classes that are in the dataset but should be ignored by the model.
            class_ignore_index:
                The class ID assigned to pixels that do not belong to any of the
                classes in `classes`. If None, the model will not ignore any classes and
                always assign a class to each pixel.
            image_size:
                The size to resize images to during inference. Default is (518, 518).
            image_normalize:
                The normalization parameters for images. Default uses ImageNet stats.
            backbone_weights:
                The path to the backbone weights. The weights must be exported
                using LightlyTrain.
            backbone_args:
                Additional arguments to pass to the backbone.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        super().__init__(locals(), ignore_args={"backbone_weights", "load_weights"})
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.classes = classes
        self.class_ignore_index = class_ignore_index
        self.backbone_freeze = backbone_freeze
        self.image_size = image_size
        self.image_normalize = image_normalize

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        # An additional class is added to represent "unknown/ignored classes" if needed.
        internal_class_to_class = list(self.classes.keys())
        if self.class_ignore_index is not None:
            internal_class_to_class.append(self.class_ignore_index)

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

        # GT masks contain the raw `class_ignore_index` value (e.g. -100) for
        # ignored pixels, so we register that key directly for the legend.
        # Predictions drop the ignore channel, so this key never appears in them.
        self.included_classes: dict[int, str] = {
            internal_class_id: class_name
            for internal_class_id, class_name in enumerate(self.classes.values())
        }
        if self.class_ignore_index is not None:
            self.included_classes[self.class_ignore_index] = "ignored"

        # Disable drop path by default for DINOv2.
        args: dict[str, Any] = {}
        # if parsed_name["package_name"] == DINOV2_VIT_PACKAGE.name:
        #     args["drop_path_rate"] = 0.0
        if backbone_args is not None:
            args.update(backbone_args)

        # Build the backbone via the package registry.
        num_channels = len(self.image_normalize["mean"])
        self.backbone: ModelWrapper = package_helpers.get_wrapped_model(
            model=f"{parsed_name['package_name']}/{parsed_name['backbone_name']}",
            num_input_channels=num_channels,
            model_args=args,
            load_weights=load_weights,
        )
        embed_dim = self.backbone.feature_dim()

        # patch_size is only available on ViT backbones.
        underlying = self.backbone.get_model()
        self.patch_size: int | None = getattr(underlying, "patch_size", None)

        # TODO(Guarin, 07/25): Improve how mask tokens are handled for fine-tuning.
        # Should we drop them from the model? We disable grads here for DDP to work
        # without find_unused_parameters=True.
        if hasattr(underlying, "mask_token"):
            underlying.mask_token.requires_grad = False

        # Load the backbone weights if a path is provided.
        # TODO(Thomas,07/2026): this should be done in the package.
        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

        if self.backbone_freeze:
            self.freeze_backbone()

        self.head = Linear(embed_dim, len(self.internal_class_to_class))

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}"
            for pkg in package_helpers.list_packages()
            for name in pkg.list_model_names()
        ]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            cls.parse_model_name(model_name=model)
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def parse_model_name(cls, model_name: str) -> dict[str, str]:
        def raise_invalid_name() -> None:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available "
                f"models are: {cls.list_model_names()}."
            )

        if not model_name.endswith(f"-{cls.model_suffix}"):
            raise_invalid_name()

        backbone_name = model_name[: -len(f"-{cls.model_suffix}")]

        try:
            package_name, backbone_name = package_helpers.parse_model_name(
                backbone_name
            )
        except ValueError:
            raise_invalid_name()

        try:
            package = package_helpers.get_package(package_name)
        except ValueError:
            raise_invalid_name()

        # Canonicalize the backbone name if the package supports it.
        if hasattr(package, "parse_model_name"):
            try:
                backbone_name = package.parse_model_name(model_name=backbone_name)
            except (ValueError, KeyError):
                raise_invalid_name()

        return {
            "model_name": f"{package_name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
            "package_name": package_name,
        }

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> Tensor:
        """Returns the predicted mask for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape (C, H, W).

        Returns:
            The predicted mask as a tensor of shape (H, W). The values represent the
            class IDs as defined in the `classes` argument of your dataset. These
            classes are also stored in the `classes` attribute of the model.
            The model will always predict the pixels as one of the known classes even when
            your dataset contains ignored classes defined by the `ignore_classes` argument.
        """
        self._track_inference()
        if self.training:
            self.eval()
        x, metadata = self.preprocess_image(image)
        batch = self.preprocess_batch([x])
        raw = self.forward_backend(batch)
        return self.postprocess(raw, [metadata])[0]

    def preprocess_image(
        self, image: PathLike | PILImage | Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype

        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        x = transforms_functional.normalize(
            x,
            mean=list(self.image_normalize["mean"]),
            std=list(self.image_normalize["std"]),
        )

        # Crop size is the short side of the training image size. We resize the image
        # such that the short side of the image matches the crop size. The long side
        # therefore varies per image, so preprocessed tensors are NOT stackable across
        # the batch.
        crop_size = min(self.image_size)
        x = transforms_functional.resize(x, size=[crop_size])

        # Origins describe where each crop sits in the resized image. They are
        # computed here (image_idx=0) so `postprocess` can untile without
        # recomputing them.
        _, origins = self.tile(images=[x])
        return x, {
            "orig_h": image_h,
            "orig_w": image_w,
            "resized_h": int(x.shape[-2]),
            "resized_w": int(x.shape[-1]),
            "origins": origins,
            "num_crops": len(origins),
        }

    def preprocess_batch(  # type: ignore[override]
        self, batch: Sequence[Tensor]
    ) -> Tensor:
        # Tile each variable-size image into square crops of size
        # `min(image_size)` and stack them into a single (B, C, H, W) tensor.
        crops_list, _ = self.tile(images=list(batch))
        return torch.stack(crops_list)

    def forward_backend(self, x: Tensor) -> Tensor:
        """Forward pass that returns per-pixel logits for a batch of crops.
        Intended for inference."""
        # x is a batch of square crops with shape (B, C, H, W).
        # forward_train already upsamples back to (H, W), so no extra interpolate
        # is needed here.
        return self.forward_train(x)

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: Tensor,
        metadata: Sequence[dict[str, Any]],
    ) -> list[Tensor]:
        # raw_outputs has shape (sum N_i, K|K+1, crop_h, crop_w).
        crop_logits_per_image = torch.split(
            raw_outputs, [meta["num_crops"] for meta in metadata], dim=0
        )

        out: list[Tensor] = []
        for image_crops, meta in zip(crop_logits_per_image, metadata):
            # Untile back to the resized image size.
            logits = self.untile(
                crop_logits=image_crops,
                origins=meta["origins"],
                image_sizes=[(meta["resized_h"], meta["resized_w"])],
            )[0]  # (K|K+1, H', W')

            if self.class_ignore_index is not None:
                # Restrict to known classes only.
                logits = logits[:-1]
            logits = logits.unsqueeze(0)  # (1, K, H', W')
            logits = F.interpolate(
                logits, size=(meta["orig_h"], meta["orig_w"]), mode="bilinear"
            )
            masks = logits.argmax(dim=1)  # (1, H, W)
            masks = self.internal_class_to_class[masks]
            out.append(masks[0])

        return out

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
    ) -> list[Tensor]:
        """Returns the predicted masks for the given batch of images.

        Args:
            images:
                Sequence of input images. Each can be a path, URL, PIL image, or
                tensor of shape (C, H, W).

        Returns:
            A list of predicted masks, one per input image. Each mask is a tensor
            of shape (H, W) where the values represent the class IDs as defined in
            the `classes` argument of your dataset. The model will always predict
            the pixels as one of the known classes even when your dataset contains
            ignored classes defined by the `ignore_classes` argument.
        """
        self._track_inference()
        if self.training:
            self.eval()
        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for image in images:
            x, meta = self.preprocess_image(image)
            tensors.append(x)
            metadata.append(meta)
        batch = self.preprocess_batch(tensors)
        raw = self.forward_backend(batch)
        return self.postprocess(raw, metadata)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Function used for ONNX export
        # TODO (Simon, 05/26) This class does not seem to have an export_onnx function
        logits = self._forward_logits(x)  # (B, K|K+1, H, W), K=num_classes
        if self.class_ignore_index is not None:
            # Restrict logits to known classes only.
            logits = logits[:, :-1]  # (1, K, H, W)
        masks = logits.argmax(dim=1)  # (B, H, W)
        masks = self.internal_class_to_class[masks]
        return masks, logits

    def forward_train(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape

        # Get the patch tokens -> (B, D, H_patch, W_patch).
        features = self.backbone.forward_features(x)["features"]

        # Classify the patch tokens -> (B, K|K+1, H_patch, W_patch), K=num_classes.
        features = features.permute(0, 2, 3, 1)  # (B, H_patch, W_patch, D)
        logits: Tensor = self.head(features)  # (B, H_patch, W_patch, K|K+1)
        logits = logits.permute(0, 3, 1, 2)  # (B, K|K+1, H_patch, W_patch)

        # Up-sample to match original image/mask resolution.
        # (B, K|K+1, H, W)
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

        return logits

    # TODO(Guarin, 08/25): Move tile/until as functions to a separate utility module.
    def tile(
        self, images: list[Tensor] | Tensor
    ) -> tuple[list[Tensor], list[tuple[int, int, int, bool]]]:
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

    def _forward_logits(self, x: Tensor) -> Tensor:
        """Forward pass that returns the logits of the last layer. Intended for
        inference."""
        # x is a batch of images with shape (B, C, H, W).

        # Tiling.
        image_sizes = [img.shape[-2:] for img in x]
        crops_list, origins = self.tile(images=x)
        crops = torch.stack(crops_list)
        crop_h, crop_w = crops.shape[-2:]

        # Forward pass.
        crop_logits = self.forward_train(crops)

        # Interpolate and untile.
        crop_logits = F.interpolate(crop_logits, (crop_h, crop_w), mode="bilinear")
        logits_list = self.untile(
            crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
        )
        logits = torch.stack(logits_list)  # (B, C, H, W)
        return logits

    def load_backbone_weights(self, path: PathLike) -> None:
        """
        Load backbone weights from a checkpoint file.

        Args:
            path: path to a .pt file, e.g., exported_last.pt.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights file not found: '{path}'")

        # Load the checkpoint.
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Load the state dict into the backbone.
        missing, unexpected = self.backbone.get_model().load_state_dict(
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
