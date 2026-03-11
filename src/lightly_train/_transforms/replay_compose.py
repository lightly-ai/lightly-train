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

from albumentations import ReplayCompose as AlbumentationsReplayCompose


class ReplayCompose(AlbumentationsReplayCompose):
    def transform_batch(
        self, batch: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Applies the same transform for all items in the batch."""
        if not batch:
            return []

        # Transform first item and record replay
        transformed = self(**batch[0])
        replay = transformed.pop("replay")

        # Transform remaining items with the same transform parameters
        result = [transformed]
        for item in batch[1:]:
            result.append(ReplayCompose.replay(replay, **item))
        return result
