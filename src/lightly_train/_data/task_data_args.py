#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

from lightly_train._configs.config import PydanticConfig


class TaskDataArgs(PydanticConfig):
    def train_imgs_path(self) -> Path:
        raise NotImplementedError()

    def val_imgs_path(self) -> Path:
        raise NotImplementedError()
