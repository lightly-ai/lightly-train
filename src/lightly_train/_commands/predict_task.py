#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import torch
from lightning_fabric import Fabric
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.connector import _PRECISION_INPUT  # type: ignore[attr-defined]
from pydantic import ConfigDict
from torch import Tensor

from lightly_train import _logging, _system
from lightly_train._commands import (
    _warnings,
    common_helpers,
    predict_task_helpers,
    train_task_helpers,
)
from lightly_train._configs import validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._events import tracker
from lightly_train._task_models import task_model_helpers
from lightly_train.types import ImageFilename, PathLike

logger = logging.getLogger(__name__)


def predict_semantic_segmentation(
    *,
    out: PathLike,
    data: PathLike | Sequence[PathLike],
    model: PathLike,
    batch_size: int = 1,  # Set this to 16 as default when we add predict_batch
    num_workers: int | Literal["auto"] = "auto",
    accelerator: str | Accelerator = "auto",
    devices: int | str | list[int] = 1,
    remove_cache: bool = False,  # TODO(Yutong, 10/25): remove/improve this when re-implementing with predict_batch
    precision: _PRECISION_INPUT = "bf16-mixed",
    overwrite: bool = False,
    log_every_num_steps: int = 100,
    num_channels: int = 3,
    loader_args: dict[str, Any] | None = None,
) -> None:
    """Predict with a semantic segmentation model to generate output masks.

    Args:
        out:
            The output directory where the output masks will be stored
        data:
            The path to a directory or a list of filenames/directories from which images must be loaded
        model:
            The path to a semantic segmentation checkpoint created by train_semantic_segmentation. This can either be a path to the checkpoint or a pretrained model name.
        batch_size:
            Global batch size. The batch size per device/GPU is inferred from this value
            and the number of devices and nodes.
        num_workers:
            Number of workers for the dataloader per device/GPU. 'auto' automatically
            sets the number of workers based on the available CPU cores.
        accelerator:
            Hardware accelerator. Can be one of ['cpu', 'gpu', 'mps', 'auto'].
            'auto' will automatically select the best accelerator available.
        devices:
            Number of devices/GPUs for prediction. The device type is determined by the ``accelerator``
            parameter.
        precision:
            Inference precision. Select '16-mixed' for mixed 16-bit precision, '32-true'
            for full 32-bit precision, or 'bf16-mixed' for mixed bfloat16 precision.
        overwrite:
            Overwrite the output directory if it already exists. Warning, this might
            overwrite existing files in the directory!
        log_every_num_steps:
            Log progress every `log_every_num_steps` steps.
        num_channels:
            Number of input channels of the images. Default is 3 (RGB).
        loader_args:
            Arguments for the PyTorch DataLoader. Should only be used in special cases
            as default values are automatically set. Prefer to use the `batch_size` and
            `num_workers` arguments instead. For details, see:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    config = validate.pydantic_model_validate(PredictTaskConfig, locals())
    _predict_task_from_config(config=config)


def _predict_task_from_config(config: PredictTaskConfig) -> None:
    # TODO(Igor, 11/25): Update to other tasks (object detection, instance segmentation, etc.) once they are supported.
    tracker.track_event(
        "inference_started", {"inference_type": "semantic_segmentation"}
    )

    config = validate.pydantic_model_validate(PredictTaskConfig, dict(config))
    initial_config = config.model_dump()
    # TODO(Guarin, 07/25): Validate and initialize arguments passed to Fabric properly.
    fabric = Fabric(
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
    )
    fabric.launch()
    config.accelerator = fabric.accelerator
    config.precision = fabric.strategy.precision.precision

    out_dir = predict_task_helpers.get_out_dir(
        fabric=fabric,
        out=config.out,
        overwrite=config.overwrite,
    )

    # Set up logging.
    _warnings.filter_train_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_file_logging(out_dir / "predict.log")
    _logging.set_up_filters()
    logger.info(f"Args: {train_task_helpers.pretty_format_args(args=initial_config)}")
    logger.info(f"Using output directory: '{out_dir}")

    # Log system information.
    system_information = _system.get_system_information()
    _system.log_system_information(system_information=system_information)

    # Load model from checkpoint to Fabric.
    model = task_model_helpers.load_model(
        model=config.model,
    )

    transform = predict_task_helpers.get_transform(
        model=model,
    )

    dataset = predict_task_helpers.get_dataset(
        data=config.data,
        transform=transform,
        num_channels=config.num_channels,
    )

    num_images = len(dataset)
    config.batch_size = common_helpers.get_global_batch_size(
        global_batch_size=config.batch_size,
        dataset=dataset,
        total_num_devices=fabric.world_size,
        loader_args=config.loader_args,
    )
    config.num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers,
        num_devices_per_node=fabric.world_size,
    )

    dataloader = predict_task_helpers.get_dataloader(
        fabric=fabric,
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        loader_args=config.loader_args,
    )
    resolved_config = config.model_dump()
    logger.info(
        f"Resolved Args: {train_task_helpers.pretty_format_args(args=resolved_config)}"
    )

    predict_model = fabric.setup_module(model)
    predict_model.mark_forward_method("predict")

    # TODO(Yutong, 10/25): re-implement with predict_batch
    logger.info(f"Creating predictions for {num_images} images...")
    for idx, batch in enumerate(dataloader):
        image_filename: ImageFilename = batch["filename"][0]
        image: Tensor = batch["views"][0][0]

        mask: Tensor = predict_model.predict(image)

        mask_numpy = mask.detach().cpu().numpy()
        mask_filepath = predict_task_helpers.compute_mask_filepath(
            config.out, config.data, image_filename
        )
        try:
            predict_task_helpers.save_mask(mask_numpy, mask_filepath)
        except Exception as e:
            logger.error(
                f"Could not save predicted mask for image "
                f"'{image_filename}' at '{mask_filepath}': {e}"
            )
        if idx % config.log_every_num_steps == 0:
            logger.info(f"Images {(idx + 1) * config.batch_size}/{num_images}")

        # free memory
        # TODO(Yutong, 10/25): remove/improve this when re-implementing with predict_batch
        if config.remove_cache:
            del batch, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Prediction completed.")


class PredictTaskConfig(PydanticConfig):
    out: PathLike
    data: PathLike | Sequence[PathLike]
    model: PathLike
    batch_size: int = 1  # Set this to 16 when we add predict_batch
    num_workers: int | Literal["auto"] = "auto"
    accelerator: str | Accelerator = "auto"
    devices: int | str | list[int] = 1
    remove_cache: bool = False  # TODO(Yutong, 10/25): remove/improve this when re-implementing with predict_batch
    precision: _PRECISION_INPUT = "bf16-mixed"
    overwrite: bool = False
    log_every_num_steps: int = 100
    num_channels: int = 3
    loader_args: dict[str, Any] | None = None

    # Allow arbitrary field types such as Module, Dataset, Accelerator, ...
    model_config = ConfigDict(arbitrary_types_allowed=True)
