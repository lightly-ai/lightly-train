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
        "num_channels_keep, prob_keep, error",
        [
            (-1, tuple(), "num_channels_keep must be at least 1, got -1"),
            (0, tuple(), "num_channels_keep must be at least 1, got 0"),
            (1, (-1.0, 0.0), "All probabilities in prob_keep must be non-negative"),
            (
                2,
                (1.0, 0.0),
                (
                    "At least num_channels_keep channels must have a non-zero "
                    "probability to be kept"
                ),
            ),
        ],
    )
    def test__init__(
        self, num_channels_keep: int, prob_keep: tuple[float, ...], error: str
    ) -> None:
        with pytest.raises(ValueError, match=error):
            ChannelDrop(num_channels_keep=num_channels_keep, prob_keep=prob_keep)

    @pytest.mark.parametrize(
        "num_channels_keep, prob_keep, expected_channels",
        [
            (1, (1.0, 0.0, 0.0, 0.0), (0,)),
            (1, (0.0, 1.0, 0.0, 0.0), (1,)),
            (2, (1.0, 1.0, 0.0, 0.0), (0, 1)),
            (2, (0.0, 0.0, 1.0, 1.0), (2, 3)),
            (2, (0.1, 0.2, 0.0, 0.0), (0, 1)),
            (2, (0.0, 0.1, 0.2, 0.1), None),
            (4, (1.0, 1.0, 1.0, 1.0), (0, 1, 2, 3)),
        ],
    )
    def test__call__(
        self,
        num_channels_keep: int,
        prob_keep: tuple[float, ...],
        expected_channels: tuple[int, ...] | None,
    ) -> None:
        image = np.random.randint(0, 255, size=(10, 10, 4), dtype=np.uint8)

        transform = ChannelDrop(
            num_channels_keep=num_channels_keep, prob_keep=prob_keep
        )
        result = transform(image=image)["image"]

        assert result.shape[0] == image.shape[0]  # Height unchanged
        assert result.shape[1] == image.shape[1]  # Width unchanged
        assert result.shape[2] == num_channels_keep  # Channels reduced
        if expected_channels is not None:
            for i, channel in enumerate(expected_channels):
                assert np.array_equal(result[:, :, i], image[:, :, channel]), (
                    f"Channel {i} does not match expected channel {channel}."
                )
