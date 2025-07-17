#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import pytest

from lightly_train._transforms.channel_drop import ChannelDrop


class TestChannelDrop:
    @pytest.mark.parametrize(
        "num_channels_keep, weight_drop, error",
        [
            (-1, tuple(), "num_channels_keep must be at least 1, got -1"),
            (0, tuple(), "num_channels_keep must be at least 1, got 0"),
            (1, (-1.0, 0.0), "All weights in weight_drop must be non-negative"),
            (
                1,
                (0.0, 0.0),
                (
                    "At most num_channels_keep channels can have zero weight "
                    "to guarantee they can be kept"
                ),
            ),
        ],
    )
    def test__init__(
        self, num_channels_keep: int, weight_drop: tuple[float, ...], error: str
    ) -> None:
        with pytest.raises(ValueError, match=error):
            ChannelDrop(num_channels_keep=num_channels_keep, weight_drop=weight_drop)

    @pytest.mark.parametrize(
        "num_channels, num_channels_keep, weight_drop, expected_channels",
        [
            (1, 1, (0.0, 1.0), (0,)),  # No drop
            (2, 1, (0.0, 1.0), (0,)),
            (2, 1, (1.0, 0.0), (1,)),
            (2, 1, (1.0, 1.0), (None,)),
            (3, 1, (0.0, 1.0, 1.0), (0,)),
            (2, 2, (0.0, 0.0, 1.0), (0, 1)),  # No drop
            (3, 2, (0.0, 0.0, 1.0), (0, 1)),
            (3, 2, (1.0, 1.0, 0.0), (None, 2)),
            (3, 3, (1.0, 1.0, 0.0, 0.0), (0, 1, 2)),  # No drop
            (4, 3, (1.0, 1.0, 0.0, 0.0), (None, 2, 3)),  # NRGB to NGB or RGB
            (4, 3, (0.0, 1.0, 1.0, 1.0), (0, None, None)),
            (4, 3, (0.0, 1.0, 1.0, 0.0), (0, None, 3)),
        ],
    )
    def test__call__(
        self,
        num_channels: int,
        num_channels_keep: int,
        weight_drop: tuple[float, ...],
        expected_channels: tuple[int, ...],
    ) -> None:
        image = np.random.randint(0, 255, size=(3, 3, num_channels), dtype=np.uint8)

        transform = ChannelDrop(
            num_channels_keep=num_channels_keep, weight_drop=weight_drop
        )
        result = transform(image=image)["image"]

        assert result.dtype == image.dtype  # dtype unchanged
        assert result.shape[0] == image.shape[0]  # Height unchanged
        assert result.shape[1] == image.shape[1]  # Width unchanged
        assert result.shape[2] == num_channels_keep  # Channels reduced
        assert len(expected_channels) == num_channels_keep
        for i, channel in enumerate(expected_channels):
            if channel is None:
                continue
            assert np.array_equal(result[:, :, i], image[:, :, channel]), (
                f"Channel {i} does not match expected channel {channel}."
            )
