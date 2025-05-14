#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
import logging
import torch

from pathlib import Path
from torch.nn import Module
from lightly_train._data.download import download_from_url
from lightly_train._commands.common_helpers import is_local_rank_zero, get_node_rank

logger = logging.getLogger(__name__)


def load_weights(model: Module, checkpoint_dir: Path, url: str) -> Module:
    # Create the directory if it doesn't exist.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Cache the teacher checkpoint.
    checkpoint_path = checkpoint_dir / Path(str(get_node_rank())) / Path(url).name 

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
    torch.distributed.barrier()
    # Load the checkpoint.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    logger.debug(f"Loaded teacher weights from '{checkpoint_path}'")
    return model