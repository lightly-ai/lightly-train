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
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module

from lightly_train._configs.config import PydanticConfig
from lightly_train._pre_post_processors.shared import (
    NormalizeBatchablePreProcessor,
    NormalizePreProcessorArgs,
    ResizeNonBatchablePreProcessor,
    ResizePreProcessorArgs,
)
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import PathLike


class SemanticSegmentationPreProcessorArgs(PydanticConfig):
    """Configuration for semantic segmentation pre-processors."""

    image_size: tuple[int, int]
    normalize: NormalizeArgs


class SemanticSegmentationNonBatchablePreProcessor:
    """Per-image (non-batchable) preprocessing for semantic segmentation.

    Loads the image, converts dtype, and resizes to the target size. Returns the
    preprocessed tensor and metadata with the original image dimensions, which the
    post-processor uses to restore the prediction to the input resolution.

    Normalization is intentionally deferred to the batchable step
    (SemanticSegmentationBatchablePreProcessor) since v2.Normalize supports batched
    (B, C, H, W) inputs and is therefore more efficient when applied to the full batch.
    """

    def __init__(self, args: SemanticSegmentationPreProcessorArgs) -> None:
        self._resize = ResizeNonBatchablePreProcessor(
            ResizePreProcessorArgs(image_size=args.image_size)
        )

    def __call__(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, dict[str, Any]]:
        return self._resize(image, device=device, dtype=dtype)


class SemanticSegmentationBatchablePreProcessor:
    """Batch-level (batchable) preprocessing for semantic segmentation.

    Stacks a list of same-size preprocessed tensors into a single (B, C, H, W) batch
    tensor and normalizes. Being uniform GPU-friendly operations, these steps can be
    baked into an exported graph (ONNX / TensorRT).
    """

    def __init__(self, args: SemanticSegmentationPreProcessorArgs) -> None:
        self._normalize = NormalizeBatchablePreProcessor(
            NormalizePreProcessorArgs(normalize=args.normalize)
        )

    def __call__(self, images: list[Tensor]) -> Tensor:
        return self._normalize(images)


class SemanticSegmentationPostProcessorArgs(PydanticConfig):
    """Configuration for the semantic segmentation post-processor."""

    pass


class SemanticSegmentationPostProcessor(Module):
    """Post-processor for semantic segmentation.

    Drops the ignore-class channel if present, applies argmax, and maps internal
    contiguous class indices to the user-facing class IDs.

    Interpolation back to the original image resolution is intentionally NOT performed
    here — it is the caller's responsibility to upsample logits to the desired output
    size before calling this module. This keeps the post-processor architecture-agnostic
    (e.g. U-Net outputs already match the input resolution and need no upsampling).

    Implemented as an nn.Module so the internal_class_to_class buffer is automatically
    moved to the correct device together with the parent model.
    """

    internal_class_to_class: Tensor

    def __init__(
        self,
        internal_class_to_class: Tensor,
        class_ignore_index: int | None,
    ) -> None:
        super().__init__()
        self.class_ignore_index = class_ignore_index
        self.register_buffer(
            "internal_class_to_class",
            internal_class_to_class,
            persistent=False,
        )

    def forward(self, raw_outputs: Tensor) -> list[Tensor]:
        """
        Args:
            raw_outputs:
                Per-pixel logits of shape (B, K|K+1, H, W). The spatial dimensions
                must already match the desired output resolution — no upsampling is
                applied here.

        Returns:
            A list of predicted masks, one per image. Each mask is a tensor of shape
            (H, W) holding user-facing class IDs.
        """
        if self.class_ignore_index is not None:
            raw_outputs = raw_outputs[:, :-1]  # drop ignore class channel -> (B, K, H, W)
        masks = raw_outputs.argmax(dim=1)  # (B, H, W)
        masks = self.internal_class_to_class[masks]
        return list(masks.unbind(0))  # list of (H, W)
