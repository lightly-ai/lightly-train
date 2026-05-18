#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from lightning_fabric.loggers.logger import Logger as FabricLogger
from PIL.Image import Image as PILImage

from lightly_train._loggers import logger_helpers
from lightly_train._task_models.train_model import TaskStepResult

_IMAGE_EXAMPLES_DIR = "image_examples"


def save_train_step_visualizations(
    *,
    result: TaskStepResult,
    out_dir: Path,
    step: int,
    loggers: Iterable[FabricLogger],
) -> None:
    image = result.create_label_image()
    _save_and_log(
        image=image,
        path=_image_examples_dir(out_dir) / f"train_labels_{step}.jpg",
        loggers=loggers,
        key="train/labels",
        step=step,
    )


def save_val_step_visualizations(
    *,
    result: TaskStepResult,
    out_dir: Path,
    val_step: int,
    global_step: int,
    loggers: Iterable[FabricLogger],
) -> None:
    viz_dir = _image_examples_dir(out_dir)
    _save_and_log(
        image=result.create_prediction_image(),
        path=viz_dir / f"val_predictions_{val_step}.jpg",
        loggers=loggers,
        key=f"val/predictions_{val_step}",
        step=global_step,
    )
    label_path = viz_dir / f"val_labels_{val_step}.jpg"
    if not label_path.exists():
        _save_and_log(
            image=result.create_label_image(),
            path=label_path,
            loggers=loggers,
            key=f"val/labels_{val_step}",
            step=global_step,
        )


def _image_examples_dir(out_dir: Path) -> Path:
    return out_dir / _IMAGE_EXAMPLES_DIR


def _save_and_log(
    *,
    image: PILImage | None,
    path: Path,
    loggers: Iterable[FabricLogger],
    key: str,
    step: int,
) -> None:
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    logger_helpers.log_image_to_loggers(
        loggers=loggers, key=key, image=image, step=step
    )
