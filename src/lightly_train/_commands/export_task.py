#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Generator
from typing import Any, Literal

import torch
from torch import distributed

from lightly_train import _logging
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models import task_model_helpers
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


_PRECALCULATE_FOR_ONNX_EXPORT = contextvars.ContextVar(
    "PRECALCULATE_FOR_ONNX_EXPORT", default=False
)


def is_in_precalculate_for_onnx_export() -> bool:
    return _PRECALCULATE_FOR_ONNX_EXPORT.get()


@contextlib.contextmanager
def precalculate_for_onnx_export() -> Generator[None, Any, None]:
    token = _PRECALCULATE_FOR_ONNX_EXPORT.set(True)
    try:
        yield
    finally:
        _PRECALCULATE_FOR_ONNX_EXPORT.reset(token)


def export_onnx(
    *,
    out: PathLike,
    checkpoint: PathLike,
    batch_size: int = 1,
    num_channels: int = 3,
    height: int = 224,
    width: int = 224,
    overwrite: bool = False,
    format_args: dict[str, Any] | None = None,
) -> None:
    return _export_task(format="onnx", **locals())


def _export_task(
    *,
    out: PathLike,
    checkpoint: PathLike,
    format: Literal["onnx"],
    batch_size: int = 1,
    num_channels: int = 3,
    height: int = 224,
    width: int = 224,
    overwrite: bool = False,
    format_args: dict[str, Any] | None = None,
) -> None:
    """Export a model from a checkpoint.

    Args:
        out:
            Path where the exported model will be saved.
        checkpoint:
            Path to the LightlyTrain checkpoint file to export the model from.
        format:
            Format to save the model in.
        batch_size:
            Batch size of the input tensor.
        num_channels:
            Number of channels in input tensor.
        height:
            Height of the input tensor.
        width:
            Width of the input tensor.
        overwrite:
            Overwrite the output file if it already exists.
        format_args:
            Format specific arguments. Eg. "dynamic" for onnx and int8 precision for tensorrt.
    """
    config = ExportTaskConfig(**locals())
    _export_task_from_config(config=config)


def _export_task_from_config(config: ExportTaskConfig) -> None:
    # Only export on rank 0.
    if distributed.is_initialized() and distributed.get_rank() > 0:
        return

    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    out_path = common_helpers.get_out_path(
        out=config.out, overwrite=config.overwrite
    ).as_posix()  # TODO(Yutong, 07/25): make sure the format corrsponds to the output file extension!
    checkpoint_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)
    task_model = task_model_helpers.load_model_from_checkpoint(
        checkpoint=checkpoint_path
    )
    task_model.eval()

    # Export the model to ONNX format
    # TODO(Yutong, 07/25): support more formats (may use ONNX as the intermediate format)
    if config.format == "onnx":
        dummy_input = torch.randn(
            config.batch_size,
            config.num_channels,
            config.height,
            config.width,
            requires_grad=False,
        )
        with precalculate_for_onnx_export():
            task_model(dummy_input)
        logger.info(f"Exporting ONNX model to '{out_path}'")
        torch.onnx.export(
            task_model,
            (dummy_input,),
            out_path,
            input_names=["input"],
            output_names=["masks", "logits"],
            **config.format_args if config.format_args else {},
        )
    else:
        raise ValueError(
            f"Unsupported format: {config.format}. Supported formats: 'onnx'."
        )


class ExportTaskConfig(PydanticConfig):
    out: PathLike
    checkpoint: PathLike
    format: Literal["onnx"]
    batch_size: int = 1
    num_channels: int = 3
    height: int = 224
    width: int = 224
    overwrite: bool = False
    format_args: dict[str, Any] | None = (
        None  # TODO(Yutong, 07/25): use Pydantic models for format_args if needed
    )
