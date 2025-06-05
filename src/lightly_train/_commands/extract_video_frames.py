#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from lightly.data._helpers import VIDEO_EXTENSIONS
from omegaconf import DictConfig
from tqdm import tqdm

from lightly_train import _logging
from lightly_train._commands import common_helpers
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

FFMPEG_INSTALLATION_EXAMPLES = [
    "Ubuntu: 'sudo apt-get install ffmpeg'",
    "Mac: 'brew install ffmpeg'",
    "Other: visit https://ffmpeg.org/download.html",
]


def ffmpeg_is_installed() -> bool:
    logger.debug("Checking if ffmpeg is installed.")
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE)
        logger.debug("ffmpeg is installed.")
        return True
    except FileNotFoundError:
        logger.debug("Did not find ffmpeg installation.")
        return False


def assert_ffmpeg_is_installed() -> None:
    if not ffmpeg_is_installed():
        raise RuntimeError(
            f"ffmpeg is not installed! Please install using one of the following "
            f"options: {', '.join(FFMPEG_INSTALLATION_EXAMPLES)}"
        )


def extract_video_frames(
    data: PathLike,
    out: PathLike,
    overwrite: bool = False,
    frame_filename_format: str = "%09d.jpg",
    num_workers: int | Literal["auto"] = "auto",
) -> None:
    """
    Extract frames from videos using ffmpeg.

    Args:
        data:
            Path to a directory containing video files.
        out:
            Output directory to save the extracted frames.
        overwrite:
            If True, existing frames are overwritten.
        frame_filename_format:
            Filename format for the extracted frames, passed as it is to ffmpeg.
        num_workers:
            Number of parallel calls to ffmpeg. 'auto' automatically sets the number of
            workers based on the available CPU cores.

    """
    config = ExtractVideoFramesConfig(**locals())
    extract_video_frames_from_config(config=config)


def extract_video_frames_from_config(config: ExtractVideoFramesConfig) -> None:
    # Set up logging.
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")
    logger.info(f"Extracting frames from videos in '{config.data}'.")

    assert_ffmpeg_is_installed()

    num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers, num_devices_per_node=1
    )
    logger.debug(f"Using {num_workers} workers to extract frames.")
    out_dir = common_helpers.get_out_dir(
        out=config.out, resume_interrupted=False, overwrite=config.overwrite
    )
    logger.info(f"Saving extracted frames to '{out_dir}'.")

    video_files = list(Path(config.data).rglob("*"))
    video_files = [f for f in video_files if f.suffix in VIDEO_EXTENSIONS]
    logger.debug(f"Found {len(video_files)} video files in '{config.data}'.")
    if num_workers is not None:
        num_workers = max(min(len(video_files), num_workers), 1)
    logger.info(
        f"Extracting frames from {len(video_files)} videos with {num_workers} calls to ffmpeg in parallel."
    )
    with ThreadPoolExecutor(max_workers=num_workers) as thread_pool:
        for _ in tqdm(
            thread_pool.map(
                lambda video_file: _extract_video(
                    video_path=video_file,
                    out=out_dir,
                    frame_filename_format=config.frame_filename_format,
                ),
                video_files,
            ),
            total=len(video_files),
            unit="videos",
        ):
            pass


def extract_video_frames_from_dictconfig(config: DictConfig) -> None:
    logger.debug(f"Extracting video frames with config: {config}")
    config_dict = omegaconf_utils.config_to_dict(config=config)
    extract_cfg = validate.pydantic_model_validate(
        ExtractVideoFramesConfig, config_dict
    )
    extract_video_frames_from_config(config=extract_cfg)


class ExtractVideoFramesConfig(PydanticConfig):
    data: PathLike
    out: PathLike
    overwrite: bool = False
    frame_filename_format: str = "%09d.jpg"
    num_workers: int | Literal["auto"] = "auto"


class CLIExtractVideoFramesConfig(ExtractVideoFramesConfig):
    data: str
    out: str


def _extract_video(video_path: Path, out: Path, frame_filename_format: str) -> None:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path:
            Path to the video file.
        out:
            Output directory to save the extracted frames.
        frame_filename_format:
            Filename format for the extracted frames.

    """
    logger.debug(f"Extracting frames from '{video_path}'.")
    out.mkdir(parents=True, exist_ok=True)
    video_output_dir = out / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving extracted frames to '{video_output_dir}'.")
    frame_path = video_output_dir / frame_filename_format
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        str(frame_path),
    ]
    try:
        logger.debug(f"Running ffmpeg command: {cmd}")
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Error extracting frames from '{video_path}': {e.stderr.decode('utf-8')}"
        )
