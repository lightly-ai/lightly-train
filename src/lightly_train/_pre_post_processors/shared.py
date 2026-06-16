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
from torchvision.transforms.v2 import Normalize, Resize
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import PathLike


class ResizePreProcessorArgs(PydanticConfig):
    image_size: tuple[int, int]


class NormalizePreProcessorArgs(PydanticConfig):
    normalize: NormalizeArgs


class ResizeNonBatchablePreProcessor:
    """Load image → to_dtype → Resize. Returns (tensor, {"orig_h", "orig_w"}).

    to_dtype stays a functional call because dtype is only known at call time
    (it comes from the model's current dtype, e.g. float16 in mixed precision).
    """

    def __init__(self, args: ResizePreProcessorArgs) -> None:
        self.resize = Resize(list(args.image_size))

    def __call__(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, dict[str, Any]]:
        x = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = int(x.shape[-2]), int(x.shape[-1])
        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        x = self.resize(x)
        return x, {"orig_h": orig_h, "orig_w": orig_w}


class NormalizeBatchablePreProcessor:
    """Stack list[Tensor] → (B, C, H, W) then normalize.

    v2.Normalize operates on both (C, H, W) and (B, C, H, W) tensors, so
    normalization is deferred to this batched step for efficiency.
    """

    def __init__(self, args: NormalizePreProcessorArgs) -> None:
        self.normalize = Normalize(
            mean=list(args.normalize.mean),
            std=list(args.normalize.std),
        )

    def __call__(self, images: list[Tensor]) -> Tensor:
        return self.normalize(torch.stack(images))
