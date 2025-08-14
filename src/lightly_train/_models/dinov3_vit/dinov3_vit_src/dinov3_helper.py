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

import torch

from lightly_train._models.dinov3_vit.dinov3_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)

logger = logging.getLogger(__name__)


def load_weights(
    model: DinoVisionTransformer, checkpoint_dir: Path, url: str | None = None
) -> DinoVisionTransformer:
    checkpoint_path = checkpoint_dir / Path(
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    logger.debug(f"Loaded teacher weights from '{checkpoint_path}'")

    return model
