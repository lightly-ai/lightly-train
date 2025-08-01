#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lightly_train._configs.config import PydanticConfig


class TaskSaveCheckpointArgs(PydanticConfig):
    save_every_num_steps: int = 1000
