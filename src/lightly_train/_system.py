#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Sequence

import torch
from torch.cuda import _CudaDeviceProperties

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DependencyInformation:
    library: str
    version: str | None


@dataclass(frozen=True)
class GitInformation:
    branch: str
    commit: str
    uncommitted_changes: str


@dataclass(frozen=True)
class SystemInformation:
    platform: str
    python_version: str
    lightly_train_version: str
    dependencies: Sequence[DependencyInformation]
    optional_dependencies: Sequence[DependencyInformation]
    cpu_count: int | None
    gpus: Sequence[_CudaDeviceProperties]
    git_info_lightly_train: GitInformation | None
    git_info_current_directory: GitInformation | None


def get_system_information() -> SystemInformation:
    version_info = sys.version_info
    return SystemInformation(
        python_version=".".join(
            str(v) for v in (version_info.major, version_info.minor, version_info.micro)
        ),
        lightly_train_version=metadata.version("lightly-train"),
        dependencies=[
            _get_dependency_information(library=library)
            for library in [
                "torch",
                "torchvision",
                "pytorch-lightning",
                "Pillow",
                "pillow-simd",
            ]
        ],
        optional_dependencies=[
            _get_dependency_information(library=library)
            for library in [
                "super-gradients",
                "timm",
                "ultralytics",
                "wandb",
            ]
        ],
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        gpus=[
            torch.cuda.get_device_properties(device)
            for device in range(torch.cuda.device_count())
        ],
        git_info_lightly_train=_get_git_information(
            directory=str(Path(__file__).parent)
        ),
        git_info_current_directory=_get_git_information(directory=os.getcwd()),
    )


def log_system_information(system_information: SystemInformation) -> None:
    # Log platform information, Python version, and LightlyTrain version.
    logger.debug(f"Platform: {system_information.platform}")
    logger.debug(f"Python: {system_information.python_version}")
    logger.debug(f"LightlyTrain: {system_information.lightly_train_version}")

    # Log git information.
    _log_git_information(
        system_information.git_info_lightly_train,
        system_information.git_info_current_directory,
    )

    # Log dependencies.
    _log_dependency_versions(
        dependencies=system_information.dependencies,
        optional_dependencies=system_information.optional_dependencies,
    )

    # Log cpu and gpu information.
    logger.debug(f"CPUs: {system_information.cpu_count}")
    logger.debug(f"GPUs: {len(system_information.gpus)}")
    for gpu in system_information.gpus:
        logger.debug(f" - {gpu.name} {gpu.major}.{gpu.minor} ({gpu.total_memory})")

    # Log environment variables.
    logger.debug("Environment variables:")
    for var in ["CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID"]:
        value = os.environ.get(var)
        if value is not None:
            logger.debug(f" - {var}: {value}")


def _log_dependency_versions(
    dependencies: Sequence[DependencyInformation],
    optional_dependencies: Sequence[DependencyInformation],
) -> None:
    logger.debug("Dependencies:")
    for dep in dependencies:
        display_version = dep.version if dep.version is not None else "x"
        logger.debug(f" - {dep.library:<20} {display_version:>12}")
    logger.debug("Optional dependencies:")
    for dep in optional_dependencies:
        display_version = dep.version if dep.version is not None else "x"
        logger.debug(f" - {dep.library:<20} {display_version:>12}")


def _get_dependency_information(library: str) -> DependencyInformation:
    try:
        return DependencyInformation(library=library, version=metadata.version(library))
    except PackageNotFoundError:
        return DependencyInformation(library=library, version=None)


def _log_git_information(
    git_info_lightly_train: GitInformation | None,
    git_info_run_directory: GitInformation | None,
) -> None:
    logger.debug("LightlyTrain Git Information:")
    if git_info_lightly_train:
        logger.debug(f" Branch: {git_info_lightly_train.branch}")
        logger.debug(f" Commit: {git_info_lightly_train.commit}")
        logger.debug(
            f" Uncommitted changes: {git_info_lightly_train.uncommitted_changes or 'None'}"
        )
    else:
        logger.debug(" LightlyTrain is not installed from a git repository.")

    logger.debug("Run directory Git Information:")
    if git_info_run_directory:
        logger.debug(f" Branch: {git_info_run_directory.branch}")
        logger.debug(f" Commit: {git_info_run_directory.commit}")
        logger.debug(
            f" Uncommitted changes: {git_info_run_directory.uncommitted_changes or 'None'}"
        )
    else:
        logger.debug(" The code is not running from a git repository.")


def _get_git_information(directory: str) -> GitInformation | None:
    """Get git branch, commit hash, and uncommitted changes status for a given directory."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=directory,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        uncommitted_changes = str(status.strip())

        return GitInformation(branch, commit, uncommitted_changes)
    except PermissionError:
        logger.debug(f"Permission denied when accessing git in '{directory}'.")
    except FileNotFoundError:
        logger.debug("Git is not installed or not available in PATH.")
    except subprocess.CalledProcessError:
        logger.debug(f"'{directory}' is not a git repository.")
    except Exception as e:
        logger.debug(
            f"Unexpected error while fetching git info from '{directory}': {e}"
        )

    return None
