#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from lightly_train._data.download import download_from_url
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)

logger = logging.getLogger(__name__)


def get_local_rank() -> int | None:
    """Get the local rank of the current process."""
    rank_keys = ("LOCAL_RANK", "SLURM_LOCALID", "JSM_NAMESPACE_LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def get_node_rank() -> int | None:
    """Get the node rank of the current process."""
    rank_keys = ("NODE_RANK", "GROUP_RANK", "SLURM_NODEID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def is_local_rank_zero() -> bool:
    """Check if the current process is running on the local rank zero."""
    local_rank = get_local_rank()
    return local_rank == 0 or local_rank is None


def load_weights(
    model: DinoVisionTransformer, checkpoint_dir: Path, url: str
) -> DinoVisionTransformer:
    # Create the directory if it doesn't exist.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Cache the teacher checkpoint. concatenate the node rank to the checkpoint path
    # to avoid overwriting the checkpoint if multiple nodes are used.
    node_rank = get_node_rank()
    if node_rank is not None:
        file_name = f"{str(node_rank)}_{str(Path(url).name)}"
    else:
        file_name = str(Path(url).name)
    checkpoint_path = checkpoint_dir / Path(file_name)

    # Only the global rank zero downloads the checkpoint.
    if is_local_rank_zero():
        if not checkpoint_path.exists():
            logger.info(
                f"Downloading teacher weights from: '{url}' and saving them to: "
                f"'{checkpoint_path}'. The cache directory location can be configured "
                "with the LIGHTLY_TRAIN_CACHE_DIR environment variable."
            )
            download_from_url(url, checkpoint_path, timeout=180.0)

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
