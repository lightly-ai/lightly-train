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
import torch
from torchvision import tv_tensors

from lightly_train._transforms.channel_drop import ChannelDrop, ChannelDropTV


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


class TestChannelDropTV:
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
            ChannelDropTV(num_channels_keep=num_channels_keep, weight_drop=weight_drop)

    def test__init__valid(self) -> None:
        transform = ChannelDropTV(num_channels_keep=3, weight_drop=(1.0, 1.0, 0.0))
        assert transform.num_channels_keep == 3
        assert transform.weight_drop == [1.0, 1.0, 0.0]

    def test_forward_keeps_all_channels(self) -> None:
        transform = ChannelDropTV(num_channels_keep=3, weight_drop=(1.0, 1.0, 1.0))
        image = tv_tensors.Image(torch.rand(3, 32, 32))
        result = transform(image)
        assert result.shape == image.shape

    def test_forward_drops_channels(self) -> None:
        transform = ChannelDropTV(num_channels_keep=2, weight_drop=(1.0, 1.0, 0.0))
        image = tv_tensors.Image(torch.rand(3, 32, 32))
        result = transform(image)
        assert result.shape[0] == 2

    def test_forward_zero_weight_never_dropped(self) -> None:
        transform = ChannelDropTV(num_channels_keep=2, weight_drop=(0.0, 1.0, 1.0))
        image = tv_tensors.Image(torch.rand(3, 32, 32))
        result = transform(image)
        assert result.shape[0] == 2
        assert torch.equal(result[0], image[0])

    def test_forward_single_channel_returns_same(self) -> None:
        transform = ChannelDropTV(num_channels_keep=1, weight_drop=(0.0, 1.0))
        image = tv_tensors.Image(torch.rand(1, 32, 32))
        result = transform(image)
        assert torch.equal(result, image)

    def test_forward_with_four_channels(self) -> None:
        transform = ChannelDropTV(num_channels_keep=3, weight_drop=(1.0, 1.0, 0.0, 0.0))
        image = tv_tensors.Image(torch.rand(4, 32, 32))
        result = transform(image)
        assert result.shape[0] == 3

    def test_forward_multiple_iterations(self) -> None:
        transform = ChannelDropTV(num_channels_keep=2, weight_drop=(1.0, 1.0, 0.0))
        image = tv_tensors.Image(torch.rand(3, 32, 32))
        for _ in range(10):
            result = transform(image)
            assert result.shape[0] == 2

    def test_forward_non_image_returns_same(self) -> None:
        transform = ChannelDropTV(num_channels_keep=2, weight_drop=(1.0, 1.0, 0.0))
        tensor = torch.rand(3, 32, 32)
        result = transform(tensor)
        assert torch.equal(result, tensor)
