#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import io
from collections.abc import Sequence
from functools import lru_cache

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from lightly_train._task_models.image_classification.task_model import (
    ImageClassification,
)
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train_api.settings import get_settings


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def get_encoder() -> ImageClassification:
    """Returns the shared frozen backbone.

    The class head of the returned model is unused, per-user heads live in the database.
    """
    settings = get_settings()
    model = ImageClassification(
        model=settings.model_name,
        classes={0: "_unused"},
        classification_task="multiclass",
        image_size=(settings.image_size, settings.image_size),
        image_normalize=NormalizeArgs().model_dump(),
        backbone_freeze=True,
    )
    return model.to(resolve_device(settings.device)).eval()


def feature_dim() -> int:
    return int(get_encoder().backbone.feature_dim())


def decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode(images: Sequence[Image.Image]) -> Tensor:
    """Returns the (B, feature_dim) pooled features of the frozen backbone."""
    model = get_encoder()
    batch = torch.stack([model.preprocess_image(image)[0] for image in images])
    batch = model.preprocess_batch(batch)
    with torch.inference_mode():
        features = model.backbone.forward_pool(model.backbone.forward_features(batch))
    return features["pooled_features"].flatten(start_dim=1).float().cpu().clone()


def normalize_features(features: Tensor) -> Tensor:
    """L2 normalization applied before both training and prediction."""
    return F.normalize(features, dim=-1)


def feature_to_blob(feature: Tensor) -> bytes:
    return feature.detach().to(torch.float32).contiguous().numpy().tobytes()


def blob_to_feature(blob: bytes) -> Tensor:
    return torch.frombuffer(bytearray(blob), dtype=torch.float32)
