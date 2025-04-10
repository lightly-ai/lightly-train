#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from enum import Enum


class EmbeddingFormat(Enum):
    CSV = "csv"
    LIGHTLY_CSV = "lightly_csv"
    TORCH = "torch"
