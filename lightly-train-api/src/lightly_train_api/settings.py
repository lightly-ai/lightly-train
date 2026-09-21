#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIGHTLY_TRAIN_API_", protected_namespaces=()
    )

    db_path: str = "lightly_train_api.db"

    model_name: str = "dinov3/vitt16"
    image_size: int = 224
    device: str = "auto"

    train_steps: int = 1000
    train_lr: float = 1e-1
    train_weight_decay: float = 0.0
    train_max_seconds: float = 5.0

    # If False, retraining runs inline instead of going through Hatchet.
    use_hatchet: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
