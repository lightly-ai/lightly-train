#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from albumentations import BaseCompose, ReplayCompose


def transform_batch(
    transform: BaseCompose, batch: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [transform(**item) for item in batch]


def transform_replay_batch(
    transform: ReplayCompose, batch: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not batch:
        return []

    # Transform first item and record replay
    transformed = transform(**batch[0])
    replay = transformed.pop("replay")

    # Transform remaining items with the same transform parameters
    result = [transformed]
    for item in batch[1:]:
        result.append(ReplayCompose.replay(replay, **item))
    return result
