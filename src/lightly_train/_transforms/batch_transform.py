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

from albumentations import BaseCompose, BboxParams, KeypointParams, ReplayCompose
from albumentations.core.composition import TransformsSeqType


class BatchTransform:
    def __init__(self, transform: BaseCompose):
        self.transform = transform

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [self.transform(**item) for item in batch]


class BatchReplayCompose:
    """Identical to albumentations.ReplayCompose but takes batches as input and replays
    the transform with the same parameters to all items in a batch.
    """

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        save_key: str = "replay",
    ):
        self.transform = ReplayCompose(
            transforms=transforms,
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
            additional_targets=additional_targets,
            p=p,
            is_check_shapes=is_check_shapes,
            save_key=save_key,
        )

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not batch:
            return []

        # Transform first item and record replay
        transformed = self.transform(**batch[0])
        replay = transformed.pop(self.transform.save_key)

        # Transform remaining items with the same transform parameters
        result = [transformed]
        for item in batch[1:]:
            transformed = ReplayCompose.replay(replay, **item)
            transformed.pop(self.transform.save_key, None)
            result.append(transformed)
        return result
