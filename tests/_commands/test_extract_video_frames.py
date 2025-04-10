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

from lightly_train._commands import extract_video_frames
from lightly_train._commands.extract_video_frames import (
    CLIExtractVideoFramesConfig,
    ExtractVideoFramesConfig,
)

from .. import helpers

if not extract_video_frames.ffmpeg_is_installed():
    pytest.skip("ffmpeg is not installed.", allow_module_level=True)


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("out_nonempty", [False, True])
@pytest.mark.parametrize("num_workers", [2, "auto"])
def test_extract_video_frames(
    tmp_path: Path,
    overwrite: bool,
    out_nonempty: bool,
    num_workers: int | Literal["auto"],
) -> None:
    out = tmp_path / "extracted_frames"
    data = tmp_path / "videos"
    out.mkdir(parents=True, exist_ok=True)
    if out_nonempty:
        (out / "file.txt").touch()

    n_videos = 4
    n_frames_per_video = 10
    helpers.create_videos(
        videos_dir=data, n_videos=n_videos, n_frames_per_video=n_frames_per_video
    )

    if not overwrite and out_nonempty:
        with pytest.raises(ValueError):
            extract_video_frames.extract_video_frames(
                out=out,
                data=data,
                overwrite=overwrite,
                num_workers=num_workers,
            )
        return

    extract_video_frames.extract_video_frames(
        out=out,
        data=data,
        overwrite=overwrite,
        num_workers=num_workers,
    )

    for i in range(n_videos):
        assert (out / f"video_{i}").exists()
        # Check that all frames and only these frames are extracted
        assert len(list((out / f"video_{i}").iterdir())) == n_frames_per_video
        for j in range(1, n_frames_per_video + 1):
            assert (out / f"video_{i}" / f"{j:09d}.jpg").exists()


def test_extract_video_frames__custom_frame_filename_format(tmp_path: Path) -> None:
    out = tmp_path / "extracted_frames"
    data = tmp_path / "videos"
    out.mkdir(parents=True, exist_ok=True)

    n_videos = 4
    n_frames_per_video = 10
    helpers.create_videos(
        videos_dir=data, n_videos=n_videos, n_frames_per_video=n_frames_per_video
    )

    extract_video_frames.extract_video_frames(
        out=out,
        data=data,
        num_workers=2,
        frame_filename_format="%04d.png",
    )

    for i in range(n_videos):
        assert (out / f"video_{i}").exists()
        # Check that all frames and only these frames are extracted
        assert len(list((out / f"video_{i}").iterdir())) == n_frames_per_video
        for j in range(1, n_frames_per_video + 1):
            assert (out / f"video_{i}" / f"{j:04d}.png").exists()


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires Python 3.10 or higher for typing."
)
def test_extract_video_frames__parameters() -> None:
    """Tests that extract function and configs have the same parameters and default
    values.

    This test is here to make sure we don't forget to update the parameters in all
    places.
    """
    helpers.assert_same_params(
        a=ExtractVideoFramesConfig, b=extract_video_frames.extract_video_frames
    )
    helpers.assert_same_params(
        a=ExtractVideoFramesConfig, b=CLIExtractVideoFramesConfig, assert_type=False
    )
