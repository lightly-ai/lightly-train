#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingEventInfo:
    """Information for tracking training events."""

    method: str
    model: str
    epochs: int
    batch_size: int
    devices: int
