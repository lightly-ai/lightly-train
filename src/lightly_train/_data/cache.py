#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path


def get_cache_dir() -> Path:
    """Returns the cache directory for lightly-train, allowing override via env variable."""
    # Get the cache directory from the environment variable if set.
    env_dir = os.getenv("LIGHTLY_TRAIN_CACHE_DIR")
    if env_dir:
        cache_dir = Path(env_dir).expanduser().resolve()
    else:
        cache_dir = Path.home() / ".cache" / "lightly-train"

    # Create the directory if it doesn't exist.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
