#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from torch import distributed

from lightly_train import _logging
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def export_task(
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
    """Export a task model from a checkpoint.

    Args:
        out:
            Path where the exported model will be saved.
        checkpoint:
            Path to the LightlyTrain checkpoint file to export the model from.
        format:
            Format to save the model in.
        batch_size:
            Batch size for the dummy input tensor used for exporting the model.
        num_channels:
            Number of channels in the dummy input tensor.
        height:
            Height of the dummy input tensor.
        width:
            Width of the dummy input tensor.
        overwrite:
            Overwrite the output file if it already exists.
        format_args:
            Format specific arguments. Eg. "dynamic" for onnx and int8 precision for tensorrt.
    """
    config = ExportTaskConfig(**locals())
    export_task_from_config(config=config)


def export_task_from_config(config: ExportTaskConfig) -> None:
    # Only export on rank 0.
    if distributed.is_initialized() and distributed.get_rank() > 0:
        return

    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    out_path = common_helpers.get_out_path(out=config.out, overwrite=config.overwrite)
    checkpoint_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)

    # Load the model
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(checkpoint_path)
    model = DINOv2SemanticSegmentation(
        model_name="vits14",  # TODO(Yutong, 07/25): use checkpoint["model_name"],
        num_classes=2,  # TODOO(Yutong, 07/25): use checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    # Export the model to ONNX format
    # TODO(Yutong, 07/25): support more formats (may use ONNX as the intermediate format)
    if config.format == "onnx":
        dummy_input = torch.randn(
            1, config.num_channels, config.height, config.width, requires_grad=False
        )
        logger.info(f"Exporting ONNX model to '{out_path.as_posix()}'")
        torch.onnx.export(
            model,
            (dummy_input,),
            out_path.as_posix(),
            input_names=["input"],
            output_names=["mask", "logits"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "mask": {0: "batch_size", 2: "height", 3: "width"},
                "logits": {0: "batch_size", 2: "height", 3: "width"},
            },
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
