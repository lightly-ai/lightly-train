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

from lightning_fabric import Fabric
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.connector import _PRECISION_INPUT  # type: ignore[attr-defined]
from lightning_fabric.strategies.strategy import Strategy
from pydantic import ConfigDict

from lightly_train import _float32_matmul_precision, _logging, _system
from lightly_train._commands import _warnings, common_helpers
from lightly_train._commands import train_task_helpers as helpers
from lightly_train._configs import validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._configs.validate import no_auto
from lightly_train._data.infinite_cycle_iterator import InfiniteCycleIterator
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._loggers.task_logger_args import TaskLoggerArgs
from lightly_train._task_models.task_train_model import TaskTrainModelArgs
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def train_task(
    *,
    out: PathLike,
    data: dict[str, Any],
    model: str,
    task: Literal["semantic_segmentation"],
    steps: int | Literal["auto"] = "auto",
    batch_size: int | Literal["auto"] = "auto",
    num_workers: int | Literal["auto"] = "auto",
    devices: int | str | list[int] = "auto",
    num_nodes: int = 1,
    accelerator: str = "auto",
    strategy: str = "auto",
    precision: _PRECISION_INPUT = "bf16-mixed",
    float32_matmul_precision: Literal["auto", "highest", "high", "medium"] = "auto",
    overwrite: bool = False,
    resume_interrupted: bool = False,
    seed: int | None = 0,
    logger_args: dict[str, Any] | None = None,
    task_args: dict[str, Any] | None = None,
    loader_args: dict[str, Any] | None = None,
) -> None:
    config = validate.pydantic_model_validate(TrainTaskConfig, locals())
    train_task_from_config(config=config)


def train_task_from_config(config: TrainTaskConfig) -> None:
    config = validate.pydantic_model_validate(TrainTaskConfig, dict(config))
    initial_config = config.model_dump()
    # NOTE(Guarin, 07/25): We add callbacks and loggers later to fabric because we first
    # have to initialize the output directory and some other things. Fabric doesn't
    # expose a method to add callbacks and loggers later but it should be safe to do
    # so anyways.
    # TODO(Guarin, 07/25): Validate and initialize arguments passed to Fabric properly.
    fabric = Fabric(
        accelerator=config.accelerator,
        strategy=config.strategy,
        devices=config.devices,
        num_nodes=config.num_nodes,
        precision=config.precision,
    )
    fabric.launch()
    config.accelerator = fabric.accelerator
    config.strategy = fabric.strategy
    # TODO(Guarin, 07/25): Figure out how to retrieve devices from fabric.
    if config.devices == "auto":
        config.devices = fabric.world_size // config.num_nodes
    config.precision = fabric.strategy.precision.precision

    out_dir = helpers.get_out_dir(
        fabric=fabric,
        out=config.out,
        resume_interrupted=config.resume_interrupted,
        overwrite=config.overwrite,
    )

    # Set up logging.
    _warnings.filter_train_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_file_logging(out_dir / "train.log")
    _logging.set_up_filters()
    logger.info(f"Args: {helpers.pretty_format_args(args=initial_config)}")
    logger.info(f"Using output directory: '{out_dir}")

    # Log system information.
    system_information = _system.get_system_information()
    _system.log_system_information(system_information=system_information)

    fabric.seed_everything(seed=config.seed, workers=True)

    config.float32_matmul_precision = (
        _float32_matmul_precision.get_float32_matmul_precision(
            float32_matmul_precision=config.float32_matmul_precision,
        )
    )
    config.task_args = helpers.get_task_train_model_args(task_args=config.task_args)

    # TODO(Guarin, 07/25): Verify out_dir same on all local ranks, see train.py. We can simplify this
    # here as distributed processing is already initialized with fabric.

    train_dataset = helpers.get_dataset(dataset_args=config.data.get_train_args())
    val_dataset = helpers.get_dataset(dataset_args=config.data.get_val_args())
    logger.info(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # TODO(Guarin, 07/25): Choose sensible default for steps. Based on model?
    config.steps = helpers.get_steps(steps=config.steps)
    # TODO(Guarin, 07/25): Choose sensible default for batch size. Based on model?
    config.batch_size = common_helpers.get_global_batch_size(
        global_batch_size=32 if config.batch_size == "auto" else config.batch_size,
        dataset=train_dataset,
        total_num_devices=fabric.world_size,
        loader_args=config.loader_args,
    )
    config.num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers,
        num_devices_per_node=fabric.world_size // config.num_nodes,
    )
    config.logger_args = helpers.get_logger_args(
        steps=config.steps,
        logger_args=config.logger_args,
    )

    # TODO(Guarin, 07/25): Handle auto batch_size/num_workers.
    train_dataloader = helpers.get_train_dataloader(
        fabric=fabric,
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        loader_args=config.loader_args,
    )
    # TODO(Guarin, 07/25): Different batch_size/num_workers for validation?
    val_dataloader = helpers.get_val_dataloader(
        fabric=fabric,
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        loader_args=config.loader_args,
    )
    infinite_train_dataloader = InfiniteCycleIterator(iterable=train_dataloader)

    model = helpers.get_task_train_model(task_args=config.task_args)
    optimizer = model.get_optimizer()
    model, optimizer = fabric.setup(model, optimizer)  # type: ignore[assignment]

    logger.info(
        f"Resolved Args: {helpers.pretty_format_args(args=config.model_dump())}"
    )
    logger.info(f"Starting training for {config.steps} steps...")
    for step in range(config.steps):
        batch = next(infinite_train_dataloader)
        # TODO(Guarin, 07/25): Backprop.
        # TODO(Guarin, 07/25): Log loss and metrics.
        model.train_step(fabric=fabric, batch=batch)
        if step % no_auto(config.logger_args.log_every_num_steps) == 0:
            logger.info(f"Step {step}/{config.steps}")

        # TODO(Guarin, 07/25): Validate every `val_every_num_steps` steps.
        # TODO(Guarin, 07/25): Log loss and metrics.
        if step == config.steps - 1:
            for batch in val_dataloader:
                model.val_step(fabric=fabric, batch=batch)
    logger.info("Training completed.")


class TrainTaskConfig(PydanticConfig):
    out: PathLike
    data: MaskSemanticSegmentationDataArgs
    model: str
    task: Literal["semantic_segmentation"]
    steps: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"
    num_workers: int | Literal["auto"] = "auto"
    devices: int | str | list[int] = "auto"
    num_nodes: int = 1
    accelerator: str | Accelerator = "auto"
    strategy: str | Strategy = "auto"
    precision: _PRECISION_INPUT = "bf16-mixed"
    float32_matmul_precision: Literal["auto", "highest", "high", "medium"] = "auto"
    overwrite: bool = False
    resume_interrupted: bool = False
    seed: int | None = 0
    logger_args: dict[str, Any] | TaskLoggerArgs | None = None
    task_args: dict[str, Any] | TaskTrainModelArgs | None = None
    loader_args: dict[str, Any] | None = None

    # Allow arbitrary field types such as Module, Dataset, Accelerator, ...
    model_config = ConfigDict(arbitrary_types_allowed=True)
