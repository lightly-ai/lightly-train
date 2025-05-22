#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path

import torch

from lightly_train import _distributed as distributed_helpers
from lightly_train._data.download import download_from_url
from lightly_train._env import Env
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)

logger = logging.getLogger(__name__)


def load_weights(
    model: DinoVisionTransformer, checkpoint_dir: Path, url: str
) -> DinoVisionTransformer:
    # Create the directory if it doesn't exist.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Cache the teacher checkpoint. concatenate the node rank to the checkpoint path
    # to avoid overwriting the checkpoint if multiple nodes are used.
    node_rank = distributed_helpers.get_node_rank()
    if node_rank is not None:
        file_name = f"{str(node_rank)}_{str(Path(url).name)}"
    else:
        file_name = str(Path(url).name)
    checkpoint_path = checkpoint_dir / Path(file_name)

    # Only the global rank zero downloads the checkpoint.
    if distributed_helpers.is_local_rank_zero():
        if not checkpoint_path.exists():
            logger.info(
                f"Downloading teacher weights from: '{url}' and saving them to: "
                f"'{checkpoint_path}'. The cache directory location can be configured "
                "with the LIGHTLY_TRAIN_CACHE_DIR environment variable."
            )
            timeout_sec = Env.LIGHTLY_TRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC.value
            download_from_url(url, checkpoint_path, timeout=timeout_sec)

        else:
            logger.info(f"Using cached teacher weights from: '{checkpoint_path}'")

    # wait for the local zero ranks to finish downloading
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Load the checkpoint.
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)
        logger.debug(f"Loaded teacher weights from '{checkpoint_path}'")
    return model
