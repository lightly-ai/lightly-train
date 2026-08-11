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


def test_tile_image__pads_image_smaller_than_tile(
    small_tile_image: torch.Tensor,
) -> None:
    # 8x10 image, 16x16 tile. Upscaling to fit would produce a 16x20 frame and two tiles
    # at x=0 and x=4 -- coordinates in a frame 20 wide for an image only 10 wide, with no
    # record of the scale, which silently stretched every tile detection. Padding keeps a
    # single tile whose coordinates are original-image pixels.
    image = small_tile_image

    tiles, coordinates = tiling.tile_image(image=image, overlap=0.2, tile_size=(16, 16))

    assert tiles.shape == (1, 3, 16, 16)
    torch.testing.assert_close(coordinates, torch.tensor([[0, 0]], device=image.device))
    # The pixels are untouched: no interpolation happened.
    torch.testing.assert_close(tiles[0, :, :8, :10], image)
    assert torch.all(tiles[0, :, 8:, :] == 0)
    assert torch.all(tiles[0, :, :, 10:] == 0)


def test_tile_image__pads_only_the_dimension_smaller_than_the_tile() -> None:
    # h=4 < 16 is padded and yields a single row of tiles; w=20 >= 16 is not padded and
    # yields two, whose x coordinates stay inside the original 20 pixel width.
    image = torch.arange(3 * 4 * 20, dtype=torch.float32).reshape(3, 4, 20)

    tiles, coordinates = tiling.tile_image(image=image, overlap=0.0, tile_size=(16, 16))

    assert tiles.shape == (2, 3, 16, 16)
    torch.testing.assert_close(
        coordinates, torch.tensor([[0, 0], [4, 0]], device=image.device)
    )
    assert int(coordinates[:, 0].max()) < 20
    torch.testing.assert_close(tiles[0, :, :4, :], image[:, :, :16])
    torch.testing.assert_close(tiles[1, :, :4, :], image[:, :, 4:20])
    assert torch.all(tiles[:, :, 4:, :] == 0)


def test_tile_image__coordinates_never_leave_the_original_image() -> None:
    # The contract every caller relies on: coordinates are original-image pixels, so a
    # tile origin can never sit outside the image. Swept over aspect ratios and a
    # non-square tile, since tile_size is (h, w) while coordinates are (x, y).
    for h, w in [(8, 10), (4, 20), (20, 4), (1, 100), (100, 1), (32, 32)]:
        for tile_size in [(16, 16), (8, 20)]:
            image = torch.zeros(3, h, w)

            _, coordinates = tiling.tile_image(
                image=image, overlap=0.2, tile_size=tile_size
            )

            assert int(coordinates[:, 0].max()) < w, (h, w, tile_size)
            assert int(coordinates[:, 1].max()) < h, (h, w, tile_size)


def test_tile_image__tile_count_stays_bounded_for_extreme_aspect_ratios() -> None:
    # Upscaling a 1x100 image to fit a 64 pixel tile blew the other dimension up to 6400
    # and produced 100+ tiles, one model input row each. Padding produces two.
    tall, wide = torch.zeros(3, 100, 1), torch.zeros(3, 1, 100)

    tall_tiles, tall_coordinates = tiling.tile_image(
        image=tall, overlap=0.2, tile_size=(64, 64)
    )
    wide_tiles, wide_coordinates = tiling.tile_image(
        image=wide, overlap=0.2, tile_size=(64, 64)
    )

    assert tall_tiles.shape == (2, 3, 64, 64)
    assert wide_tiles.shape == (2, 3, 64, 64)
    torch.testing.assert_close(tall_coordinates, torch.tensor([[0, 0], [0, 36]]))
    torch.testing.assert_close(wide_coordinates, torch.tensor([[0, 0], [36, 0]]))


def test_tile_image__non_square_tile_pads_the_short_dimension(
    small_tile_image: torch.Tensor,
) -> None:
    # 8x10 image, (4, 16) tile: the height tiles several times while the width is padded
    # once. Guards the (height, width) vs (x, y) ordering.
    image = small_tile_image

    tiles, coordinates = tiling.tile_image(image=image, overlap=0.2, tile_size=(4, 16))

    assert tiles.shape[1:] == (3, 4, 16)
    # One column of tiles, since the padded width holds exactly one tile.
    assert int(coordinates[:, 0].max()) == 0
    assert int(coordinates[:, 1].max()) == 4
    torch.testing.assert_close(tiles[0, :, :, :10], image[:, :4, :])
    assert torch.all(tiles[:, :, :, 10:] == 0)


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
