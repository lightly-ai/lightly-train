#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

from omegaconf import DictConfig
from torch import distributed
from torch.nn import Module

from lightly_train import _logging
from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import _warnings, common_helpers
from lightly_train._commands.common_helpers import ModelFormat, ModelPart
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._models import package_helpers
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def export(
    *,
    out: PathLike,
    checkpoint: PathLike,
    part: str | ModelPart = "model",
    format: str | ModelFormat = "package_default",
    overwrite: bool = False,
) -> None:
    """Export a model from a checkpoint.

    See the documentation for more information: https://docs.lightly.ai/train/stable/export.html

    Args:
        out:
            Path where the exported model will be saved.
        checkpoint:
            Path to the LightlyTrain checkpoint file to export the model from. The
            location of the checkpoint depends on the train command. If training was run
            with ``out="out/my_experiment"``, then the last LightlyTrain checkpoint is
            saved to ``out/my_experiment/checkpoints/last.ckpt``.
        part:
            Part of the model to export. Valid options are 'model' and
            'embedding_model'. 'model' is the default option and exports the model
            that was passed as ``model`` argument to the train function.
            'embedding_model' exports the embedding model. This includes the model
            passed with the model argument in the train function and an extra embedding
            layer if the ``embed_dim`` argument was set during training. This is useful
            if you want to use the exported model for embedding images.
        format:
            Format to save the model in. Valid options are ['package_default',
            'torch_model', 'torch_state_dict'].
            'package_default' is the default option and exports the model in the
            default format of the package that was used for training. This ensures
            compatibility with the package and is the most flexible option.
            'torch_state_dict' exports the model's state dict which can be loaded with
            `model.load_state_dict(torch.load(out, weights_only=True))`.
            'torch_model' exports the model as a torch module which can be loaded with
            `model = torch.load(out)`. This requires that the same LightlyTrain version
            is installed when the model is exported and when it is loaded again.
        overwrite:
            Overwrite the output file if it already exists.
    """
    config = ExportConfig(**locals())
    export_from_config(config=config)


def export_from_config(config: ExportConfig) -> None:
    # Only export on rank 0.
    if distributed.is_initialized() and distributed.get_rank() > 0:
        return

    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    part = _get_model_part(part=config.part)
    format = _get_model_format(format=config.format)
    logger.info(f"Exporting '{part}' as '{format}'.")
    _validate_model_part_and_format(part, format)
    out_path = common_helpers.get_out_path(out=config.out, overwrite=config.overwrite)
    ckpt_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)
    logger.info(f"Loading checkpoint from '{ckpt_path}'")
    ckpt = Checkpoint.from_path(checkpoint=ckpt_path)
    model: Module | ModelWrapper | EmbeddingModel = _get_model(
        checkpoint=ckpt, part=part
    )
    logger.info(f"Exporting model to '{out_path}'")
    package = package_helpers.get_package_from_model(
        model=model, include_custom=True, fallback_custom=True
    )
    common_helpers.export_model(
        model=model, format=format, out=out_path, package=package
    )


def export_from_dictconfig(config: DictConfig) -> None:
    logger.debug(f"Exporting model with config: {config}")
    config_dict = omegaconf_utils.config_to_dict(config=config)
    export_cfg = validate.pydantic_model_validate(ExportConfig, config_dict)
    export_from_config(config=export_cfg)


class ExportConfig(PydanticConfig):
    out: PathLike
    checkpoint: PathLike
    part: str | ModelPart = "model"
    format: str | ModelFormat = "package_default"
    overwrite: bool = False


class CLIExportConfig(ExportConfig):
    out: str
    checkpoint: str
    part: str = "model"
    format: str = "package_default"


def _get_model_part(part: ModelPart | str) -> ModelPart:
    logger.debug(f"Getting model part for '{part}'.")
    try:
        return ModelPart(part)
    except ValueError:
        raise ValueError(
            f"Invalid model part: '{part}'. Valid parts are: "
            f"{[p.value for p in ModelPart]}"
        )


def _get_model_format(format: ModelFormat | str) -> ModelFormat:
    logger.debug(f"Getting model format for '{format}'.")
    try:
        return ModelFormat(format)
    except ValueError:
        raise ValueError(
            f"Invalid model format: '{format}'. Valid formats are: "
            f"{[f.value for f in ModelFormat]}"
        )


def _validate_model_part_and_format(part: ModelPart, format: ModelFormat) -> None:
    if part == ModelPart.EMBEDDING_MODEL and not (
        format == ModelFormat.TORCH_STATE_DICT or format == ModelFormat.TORCH_MODEL
    ):
        raise ValueError(
            "Only format='torch_model' or format='torch_state_dict' is supported for the embedding model."
        )


def _get_model(
    checkpoint: Checkpoint, part: ModelPart
) -> Module | EmbeddingModel | ModelWrapper:
    logger.debug(f"Getting model part: '{part}' from checkpoint.")
    if part == ModelPart.MODEL:
        return checkpoint.lightly_train.models.model
    elif part == ModelPart.WRAPPED_MODEL:
        return checkpoint.lightly_train.models.wrapped_model
    elif part == ModelPart.EMBEDDING_MODEL:
        return checkpoint.lightly_train.models.embedding_model
    else:
        raise ValueError(f"Invalid model part: {part}")
