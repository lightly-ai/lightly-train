#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

import torch

from lightly_train import _logging
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def export_onnx(
    *,
    out: PathLike,
    model_name: str,
    backbone_weights: PathLike,
    overwrite: bool = False,
) -> None:
    """Export a model from a checkpoint.

    See the documentation for more information: https://docs.lightly.ai/train/stable/export.html

    Args:
        out:
            Path where the exported model will be saved.
        overwrite:
            Overwrite the output file if it already exists.
    """
    config = ExportONNXConfig(**locals())
    export_onnx_from_config(config=config)


def export_onnx_from_config(config: ExportONNXConfig) -> None:
    # Only export on rank 0.

    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    out_path = common_helpers.get_out_path(out=config.out, overwrite=config.overwrite)
    backbone_weights = common_helpers.get_checkpoint_path(
        checkpoint=config.backbone_weights
    )

    # Load the model
    torch.use_deterministic_algorithms(True)
    model = DINOv2SemanticSegmentation(
        model_name=config.model_name,
        num_classes=2,  # TODO
        backbone_weights=backbone_weights,  # TODO
    )
    model.eval()

    # Export the model to ONNX format
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
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
    )


class ExportONNXConfig(PydanticConfig):
    out: PathLike
    model_name: str
    backbone_weights: PathLike
    overwrite: bool = False
