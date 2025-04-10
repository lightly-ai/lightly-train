#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import re

import pytest
from omegaconf import OmegaConf
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture

from lightly_train import _cli
from lightly_train._commands import extract_video_frames
from lightly_train._commands.embed import CLIEmbedConfig
from lightly_train._commands.export import ExportConfig
from lightly_train._commands.extract_video_frames import ExtractVideoFramesConfig
from lightly_train._commands.train import CLITrainConfig
from lightly_train._configs.config import PydanticConfig


@pytest.mark.parametrize(
    "command,msg",
    [
        (["help"], _cli._HELP_MSG),
        (["train"], _cli._TRAIN_HELP_MSG),
        (["train", "help"], _cli._TRAIN_HELP_MSG),
        (["export"], _cli._EXPORT_HELP_MSG),
        (["export", "help"], _cli._EXPORT_HELP_MSG),
    ],
)
def test_cli__help(command: list[str], msg: str, caplog: LogCaptureFixture) -> None:
    config = OmegaConf.from_cli(command)
    with caplog.at_level(level="INFO"):
        _cli.cli(config=config)
        assert _cli._format_msg(msg) == caplog.records[0].message


def test_cli__train(mocker: MockerFixture) -> None:
    config = OmegaConf.from_cli(["train", "out=out"])
    mock_train_from_config = mocker.patch.object(_cli.train, "train_from_dictconfig")
    _cli.cli(config=config)
    mock_train_from_config.assert_called_once()
    mock_train_from_config.assert_called_once_with(config)


def test_cli__export(mocker: MockerFixture) -> None:
    config = OmegaConf.from_cli(["export", "out=model.pt"])
    mock_export_from_dictconfig = mocker.patch.object(
        _cli.export, "export_from_dictconfig"
    )
    _cli.cli(config=config)
    mock_export_from_dictconfig.assert_called_once()
    mock_export_from_dictconfig.assert_called_once_with(config)


def test_cli__embed(mocker: MockerFixture) -> None:
    config = OmegaConf.from_cli(["embed", "out=embeddings.csv"])
    mock_embed_from_config = mocker.patch.object(_cli.embed, "embed_from_dictconfig")
    _cli.cli(config=config)
    mock_embed_from_config.assert_called_once()
    mock_embed_from_config.assert_called_once_with(config)


@pytest.mark.skipif(
    not extract_video_frames.ffmpeg_is_installed(), reason="ffmpeg is not installed."
)
def test_cli__extract_video_frames(mocker: MockerFixture) -> None:
    config = OmegaConf.from_cli(["extract_video_frames", "data=videos", "out=frames"])
    mock_extract_video_frames = mocker.patch.object(
        _cli.extract_video_frames, "extract_video_frames_from_dictconfig"
    )
    _cli.cli(config=config)
    mock_extract_video_frames.assert_called_once()
    mock_extract_video_frames.assert_called_once_with(config)


def test_cli__list_models(caplog: LogCaptureFixture) -> None:
    config = OmegaConf.from_cli(["list_models"])
    with caplog.at_level(level="INFO"):
        _cli.cli(config=config)
        assert "    torchvision/resnet18" in caplog.records[0].message


def test_cli__list_methods(caplog: LogCaptureFixture) -> None:
    config = OmegaConf.from_cli(["list_methods"])
    with caplog.at_level(level="INFO"):
        _cli.cli(config=config)
        assert "    simclr" in caplog.records[0].message


def test__TRAIN_HELP_MSG__parameters() -> None:
    """Test that the train help message contains all parameters from CLITrainConfig."""
    _assert_help_msg_contains_params(
        msg=_cli._TRAIN_HELP_MSG, config=CLITrainConfig(out="", data="", model="")
    )


def test__EXPORT_HELP_MSG__parameters() -> None:
    """Test that the export help message contains all parameters from ExportConfig."""
    _assert_help_msg_contains_params(
        msg=_cli._EXPORT_HELP_MSG,
        config=ExportConfig(checkpoint="", out="", part="", format=""),
    )


def test__EMBED_HELP_MSG__parameters() -> None:
    """Test that the embed help message contains all parameters from CLIEmbedConfig."""
    _assert_help_msg_contains_params(
        msg=_cli._EMBED_HELP_MSG,
        config=CLIEmbedConfig(out="", data="", checkpoint="", format=""),
    )


def test__EXTRACT_VIDEO_FRAMES_HELP_MSG__parameters() -> None:
    """Test that the extract_video_frames help message contains all parameters from ExtractVideoFramesConfig."""
    _assert_help_msg_contains_params(
        msg=_cli._EXTRACT_VIDEO_FRAMES_HELP_MSG,
        config=ExtractVideoFramesConfig(out="", data=""),
    )


def _assert_help_msg_contains_params(msg: str, config: PydanticConfig) -> None:
    for param in config.model_dump().keys():
        assert re.search(rf"{param} \(.*\):", msg), f"{param} is missing"
