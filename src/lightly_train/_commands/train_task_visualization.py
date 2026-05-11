#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from PIL.Image import Image as PILImage

from lightly_train._task_models.train_model import TaskStepResult

_IMAGE_EXAMPLES_DIR = "image_examples"


def save_train_step_visualizations(
    *, result: TaskStepResult, out_dir: Path, step: int
) -> None:
    _save_image(
        image=result.create_label_image(),
        path=_image_examples_dir(out_dir) / f"train_labels_{step}.jpg",
    )


def save_val_step_visualizations(
    *,
    result: TaskStepResult,
    out_dir: Path,
    val_step: int,
    save_label_image: bool = False,
) -> None:
    viz_dir = _image_examples_dir(out_dir)
    _save_image(
        image=result.create_prediction_image(),
        path=viz_dir / f"val_predictions_{val_step}.jpg",
    )
    if save_label_image:
        _save_image(
            image=result.create_label_image(),
            path=viz_dir / f"val_labels_{val_step}.jpg",
        )


def _image_examples_dir(out_dir: Path) -> Path:
    return out_dir / _IMAGE_EXAMPLES_DIR


def _save_image(*, image: PILImage | None, path: Path) -> None:
    if image is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
