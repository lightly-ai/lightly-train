#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from lightly_train._env import Env


def get_model_cache_dir() -> Path:
    """Returns the model cache directory for LightlyTrain, allowing override via env
    variable.
    """
    # Get the cache directory from the environment variable if set.
    cache_dir = Env.LIGHTLY_TRAIN_MODEL_CACHE_DIR.value.expanduser().resolve()
    # Create the directory if it doesn't exist.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_data_cache_dir() -> Path:
    """Get the data cache directory for LightlyTrain."""
    # Get the cache directory from the environment variable if set.
    cache_dir = Env.LIGHTLY_TRAIN_DATA_CACHE_DIR.value.expanduser().resolve()
    # Create the directory if it doesn't exist.
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
