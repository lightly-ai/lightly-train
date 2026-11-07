#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lightly_train._pre_post_processing import tiling


@pytest.fixture
def tile_image() -> torch.Tensor:
    return torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)


@pytest.fixture
def small_tile_image() -> torch.Tensor:
    return torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(3, 8, 10)


def test_tile_image(tile_image: torch.Tensor) -> None:
    image = tile_image

    tiles, coordinates = tiling.tile_image(image=image, overlap=0.5, tile_size=(16, 16))

    assert tiles.shape == (9, 3, 16, 16)
    torch.testing.assert_close(
        coordinates,
        torch.tensor(
            [
                [0, 0],
                [8, 0],
                [16, 0],
                [0, 8],
                [8, 8],
                [16, 8],
                [0, 16],
                [8, 16],
                [16, 16],
            ],
            device=image.device,
        ),
    )
    torch.testing.assert_close(tiles[0], image[:, :16, :16])
    torch.testing.assert_close(tiles[-1], image[:, 16:32, 16:32])


def test_tile_image__resize_mode_on_small_image(small_tile_image: torch.Tensor) -> None:
    image = small_tile_image

    tiles, coordinates = tiling.tile_image(
        image=image, overlap=0.2, tile_size=(16, 16), padding_mode="resize"
    )

    assert tiles.shape == (2, 3, 16, 16)
    torch.testing.assert_close(
        coordinates,
        torch.tensor([[0, 0], [4, 0]], device=image.device),
    )
    resized_image = F.interpolate(
        image.unsqueeze(0), size=(16, 20), mode="bilinear", align_corners=False
    ).squeeze(0)
    torch.testing.assert_close(tiles[0], resized_image[:, :16, :16])
    torch.testing.assert_close(tiles[-1], resized_image[:, :16, 4:20])


def test_tile_image__pad_mode_on_small_image(small_tile_image: torch.Tensor) -> None:
    image = small_tile_image

    tiles, coordinates = tiling.tile_image(
        image=image, overlap=0.2, tile_size=(16, 16), padding_mode="pad"
    )

    assert tiles.shape == (1, 3, 16, 16)
    torch.testing.assert_close(coordinates, torch.tensor([[0, 0]], device=image.device))
    torch.testing.assert_close(tiles[0, :, :8, :10], image)
    assert torch.all(tiles[0, :, 8:, :] == 0)
    assert torch.all(tiles[0, :, :, 10:] == 0)


def test_tile_image__appends_last_tile_for_non_divisible_size() -> None:
    # 30 is not a multiple of the tile size, so the last tile in each dimension
    # must be snapped back to size - tile_size (14) to stay within the image.
    image = torch.arange(3 * 30 * 30, dtype=torch.float32).reshape(3, 30, 30)

    tiles, coordinates = tiling.tile_image(image=image, overlap=0.0, tile_size=(16, 16))

    assert tiles.shape == (4, 3, 16, 16)
    torch.testing.assert_close(
        coordinates,
        torch.tensor([[0, 0], [14, 0], [0, 14], [14, 14]], device=image.device),
    )
    torch.testing.assert_close(tiles[0], image[:, :16, :16])
    torch.testing.assert_close(tiles[-1], image[:, 14:30, 14:30])


@pytest.mark.parametrize("overlap", [-0.1, 1.0])
def test_tile_image__raises_for_invalid_overlap(
    overlap: float, tile_image: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        tiling.tile_image(image=tile_image, overlap=overlap, tile_size=(16, 16))


@pytest.mark.parametrize("tile_size", [(0, 16), (16, 0), (-1, 16), (16, -1)])
def test_tile_image__raises_for_invalid_tile_size(
    tile_size: tuple[int, int], tile_image: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="tile_size"):
        tiling.tile_image(image=tile_image, overlap=0.2, tile_size=tile_size)


def test_tile_image__raises_for_invalid_padding_mode(
    tile_image: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="padding_mode"):
        tiling.tile_image(
            image=tile_image,
            overlap=0.2,
            tile_size=(16, 16),
            padding_mode="invalid",  # type: ignore[arg-type]
        )
