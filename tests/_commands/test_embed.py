#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pytest
import torch
from omegaconf import OmegaConf
from pytest_mock import MockerFixture

from lightly_train._commands import embed
from lightly_train._commands.embed import CLIEmbedConfig, EmbedConfig
from lightly_train._embedding.embedding_format import EmbeddingFormat

from .. import helpers


def test_embed__cpu(tmp_path: Path) -> None:
    out = tmp_path / "embeddings.csv"
    data = tmp_path / "data"
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    helpers.create_images(image_dir=data, files=2)
    helpers.get_checkpoint().save(path=checkpoint)
    embed.embed(
        out=out,
        data=data,
        checkpoint=checkpoint,
        format=EmbeddingFormat.CSV,
        batch_size=1,
        accelerator="cpu",
    )
    lines = out.read_text().splitlines()
    assert lines[0] == "filename,embedding_0,embedding_1"
    filenames = sorted([line.split(",")[0] for line in lines[1:]])
    assert filenames == ["0.png", "1.png"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize("num_workers", [0, 2, "auto"])
def test_embed(tmp_path: Path, num_workers: int | Literal["auto"]) -> None:
    out = tmp_path / "embeddings.csv"
    data = tmp_path / "data"
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    helpers.create_images(image_dir=data, files=2)
    helpers.get_checkpoint().save(path=checkpoint)
    embed.embed(
        out=out,
        data=data,
        checkpoint=checkpoint,
        format=EmbeddingFormat.CSV,
        batch_size=1,
        num_workers=num_workers,
    )
    lines = out.read_text().splitlines()
    assert lines[0] == "filename,embedding_0,embedding_1"
    filenames = sorted([line.split(",")[0] for line in lines[1:]])
    assert filenames == ["0.png", "1.png"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_embed__overwrite_true(tmp_path: Path) -> None:
    out = tmp_path / "embeddings.csv"
    out.touch()
    data = tmp_path / "data"
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    helpers.create_images(image_dir=data, files=1)
    helpers.get_checkpoint().save(path=checkpoint)
    embed.embed(
        out=out,
        data=data,
        checkpoint=checkpoint,
        format=EmbeddingFormat.CSV,
        batch_size=1,
        num_workers=0,
        overwrite=True,
    )
    # Assert file was overwritten
    assert out.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_embed__varying_image_resolution(tmp_path: Path) -> None:
    out = tmp_path / "embeddings.csv"
    out.touch()

    data = tmp_path / "data"
    data.mkdir(exist_ok=True, parents=True)
    helpers.create_image(path=data / "img.jpg", width=300, height=224)
    helpers.create_image(path=data / "img_2.jpg", width=224, height=300)

    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    helpers.get_checkpoint().save(path=checkpoint)

    embed.embed(
        out=out,
        data=data,
        checkpoint=checkpoint,
        format=EmbeddingFormat.CSV,
        batch_size=2,
        num_workers=0,
        overwrite=True,
    )
    # Assert file was overwritten
    assert out.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_embed__overwrite_false(tmp_path: Path) -> None:
    out = tmp_path / "embeddings.csv"
    out.touch()
    with pytest.raises(ValueError, match=f"Output '{out}' already exists!"):
        embed.embed(
            out=out,
            data=tmp_path / "data",
            checkpoint=tmp_path / "last.ckpt",
            format=EmbeddingFormat.CSV,
            batch_size=1,
            num_workers=0,
            overwrite=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_embed__invalid_format(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        # Wildcard to match the full error message and avoid test failing everytime
        # a new format is added.
        match=(
            r"Invalid embedding format: 'invalid_format'. Valid formats are: "
            r"\['csv', 'lightly_csv', .*\]"
        ),
    ):
        embed.embed(
            out=tmp_path / "embeddings.csv",
            data=tmp_path / "data",
            checkpoint=tmp_path / "last.ckpt",
            format="invalid_format",
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires Python 3.10 or higher for typing."
)
def test_embed__parameters() -> None:
    """Tests that embed function and configs have the same parameters and default
    values.

    This test is here to make sure we don't forget to update the parameters in all
    places.
    """
    helpers.assert_same_params(a=EmbedConfig, b=embed.embed)
    helpers.assert_same_params(a=EmbedConfig, b=CLIEmbedConfig, assert_type=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_embed_from_dictconfig(tmp_path: Path) -> None:
    out = tmp_path / "embeddings.csv"
    data = tmp_path / "data"
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    helpers.create_images(image_dir=data, files=2)
    helpers.get_checkpoint().save(path=checkpoint)
    embed.embed_from_dictconfig(
        OmegaConf.create(
            {
                "out": str(out),
                "data": str(data),
                "checkpoint": str(checkpoint),
                "format": "csv",
                "num_workers": 2,
            }
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize(
    "image_size, expected",
    [
        (128, 128),
        ("[128,224]", (128, 224)),
    ],
)
def test_embed_from_dictconfig__image_size(
    image_size: int | str, expected: tuple[int, int], mocker: MockerFixture
) -> None:
    """Tests that image_size is correctly parsed"""
    config = OmegaConf.from_cli(
        [
            "out=out",
            "data=data",
            "checkpoint=checkpoint",
            f"image_size={image_size}",
            "format=csv",
        ]
    )
    mock_embed = mocker.patch.object(embed, "embed_from_config")
    embed.embed_from_dictconfig(config)
    assert mock_embed.call_args.kwargs["config"].image_size == expected


@pytest.mark.parametrize(
    "ckpt_dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
    ],
)
def test_embed_from_dictconfig__from_xbit_ckpts(
    tmp_path: Path, ckpt_dtype: torch.dtype
) -> None:
    # 1. Setup the checkpoint
    out = tmp_path / "embeddings.csv"
    data = tmp_path / "data"
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    helpers.create_images(image_dir=data, files=2)
    ckpt = helpers.get_checkpoint(dtype=ckpt_dtype)
    ckpt.save(path=checkpoint)

    embed.embed_from_dictconfig(
        OmegaConf.create(
            {
                "out": str(out),
                "data": str(data),
                "checkpoint": str(checkpoint),
                "format": "csv",
                "num_workers": 2,
            }
        )
    )
