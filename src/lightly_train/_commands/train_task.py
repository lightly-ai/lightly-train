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
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.train_model import TrainModelArgs
from lightly_train._train_task_state import TrainTaskState
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def train_semantic_segmentation(
    *,
    out: PathLike,
    data: dict[str, Any],
    model: str,
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
    model_args: dict[str, Any] | None = None,
    transform_args: dict[str, Any] | None = None,
    loader_args: dict[str, Any] | None = None,
    save_checkpoint_args: dict[str, Any] | None = None,
) -> None:
    """Train a semantic segmentation model.

    See the documentation for more information: https://docs.lightly.ai/train/stable/semantic_segmentation.html

        The training process can be monitored with TensorBoard:

    .. code-block:: bash

        tensorboard --logdir out

    After training, the last model checkpoint is saved in the out directory to:
    ``out/checkpoints/last.ckpt``.

    Args:
        out:
            The output directory where the model checkpoints and logs are saved.
        data:
            The dataset configuration. See the documentation for more information:
            https://docs.lightly.ai/train/stable/semantic_segmentation.html#data
        model:
            The model to train. For example, "dinov2/vits14-eomt".
        steps:
            The number of training steps.
        batch_size:
            Global batch size. The batch size per device/GPU is inferred from this value
            and the number of devices and nodes.
        num_workers:
            Number of workers for the dataloader per device/GPU. 'auto' automatically
            sets the number of workers based on the available CPU cores.
        devices:
            Number of devices/GPUs for training. 'auto' automatically selects all
            available devices. The device type is determined by the ``accelerator``
            parameter.
        num_nodes:
            Number of nodes for distributed training.
        accelerator:
            Hardware accelerator. Can be one of ['cpu', 'gpu', 'mps', 'auto'].
            'auto' will automatically select the best accelerator available.
        strategy:
            Training strategy. For example 'ddp' or 'auto'. 'auto' automatically
            selects the best strategy available.
        precision:
            Training precision. Select '16-mixed' for mixed 16-bit precision, '32-true'
            for full 32-bit precision, or 'bf16-mixed' for mixed bfloat16 precision.
        float32_matmul_precision:
            Precision for float32 matrix multiplication. Can be one of ['auto',
            'highest', 'high', 'medium']. See https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
            for more information.
        overwrite:
            Overwrite the output directory if it already exists. Warning, this might
            overwrite existing files in the directory!
        resume_interrupted:
            Set this to True if you want to resume training from an **interrupted or
            crashed** training run. This will pick up exactly where the training left
            off, including the optimizer state and the current step.

            - You must use the same ``out`` directory as the interrupted run.
            - You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).
            - This is intended for continuing the same run without modification.
        seed:
            Random seed for reproducibility.
        logger_args:
            Logger arguments. Either None or a dictionary of logger names to either
            None or a dictionary of logger arguments. None uses the default loggers.
            To disable a logger, set it to None: ``logger_args={"tensorboard": None}``.
            To configure a logger, pass the respective arguments:
            ``logger_args={"mlflow": {"experiment_name": "my_experiment", ...}}``.
            See https://docs.lightly.ai/train/stable/semantic_segmentation.html#logging
            for more information.
        model_args:
            Model training arguments. Either None or a dictionary of model arguments.
        transform_args:
            Transform arguments. Either None or a dictionary of transform arguments.
            The image size and normalization parameters can be set with
            ``transform_args={"image_size": (height, width), "normalize": {"mean": (r, g, b), "std": (r, g, b)}}``
        loader_args:
            Arguments for the PyTorch DataLoader. Should only be used in special cases
            as default values are automatically set. Prefer to use the `batch_size` and
            `num_workers` arguments instead. For details, see:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        save_checkpoint_args:
            Arguments to configure the saving of checkpoints. The checkpoint frequency
            can be set with ``save_checkpoint_args={"save_every_num_steps": 100}``.
    """
    return _train_task(task="semantic_segmentation", **locals())


def _train_task(
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
    model_args: dict[str, Any] | None = None,
    transform_args: dict[str, Any] | None = None,
    loader_args: dict[str, Any] | None = None,
    save_checkpoint_args: dict[str, Any] | None = None,
) -> None:
    config = validate.pydantic_model_validate(TrainTaskConfig, locals())
    _train_task_from_config(config=config)


def _train_task_from_config(config: TrainTaskConfig) -> None:
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
    config.save_checkpoint_args = helpers.get_save_checkpoint_args(
        checkpoint_args=config.save_checkpoint_args
    )
    train_model_cls = helpers.get_train_model_cls(
        model_name=config.model,
    )

    train_transform_args, val_transform_args = helpers.get_transform_args(
        train_model_cls=train_model_cls,
        transform_args=config.transform_args,
        ignore_index=config.data.ignore_index,
    )
    train_transform = helpers.get_train_transform(
        train_model_cls=train_model_cls, train_transform_args=train_transform_args
    )
    val_transform = helpers.get_val_transform(
        train_model_cls=train_model_cls, val_transform_args=val_transform_args
    )

    with helpers.get_dataset_temp_mmap_path(
        fabric=fabric, data=config.data.train.images
    ) as train_mmap_filepath, helpers.get_dataset_temp_mmap_path(
        fabric=fabric, data=config.data.val.images
    ) as val_mmap_filepath:
        train_dataset = helpers.get_dataset(
            fabric=fabric,
            dataset_args=config.data.get_train_args(),
            transform=train_transform,
            mmap_filepath=train_mmap_filepath,
        )
        val_dataset = helpers.get_dataset(
            fabric=fabric,
            dataset_args=config.data.get_val_args(),
            transform=val_transform,
            mmap_filepath=val_mmap_filepath,
        )
        logger.info(
            f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}"
        )

        train_model_args_cls = train_model_cls.train_model_args_cls

        config.steps = helpers.get_steps(
            steps=config.steps, default_steps=train_model_args_cls.default_steps
        )
        config.batch_size = common_helpers.get_global_batch_size(
            global_batch_size=(
                train_model_args_cls.default_batch_size
                if config.batch_size == "auto"
                else config.batch_size
            ),
            dataset=train_dataset,
            total_num_devices=fabric.world_size,
            loader_args=config.loader_args,
        )
        config.num_workers = common_helpers.get_num_workers(
            num_workers=config.num_workers,
            num_devices_per_node=fabric.world_size // config.num_nodes,
        )

        config.model_args = helpers.get_train_model_args(
            model_args=config.model_args,
            model_args_cls=train_model_args_cls,
            total_steps=no_auto(config.steps),
            model_name=config.model,
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

        config.logger_args = helpers.get_logger_args(
            steps=config.steps,
            val_steps=len(val_dataloader),
            logger_args=config.logger_args,
        )
        logger_instances = helpers.get_loggers(
            logger_args=config.logger_args,
            out=out_dir,
        )
        fabric.loggers.extend(logger_instances)

        train_model = train_model_cls(
            model_name=config.model,
            model_args=config.model_args,
            data_args=config.data,
            val_transform_args=val_transform_args,
        )

        # Set train mode to make sure that all parameters are in the correct state before
        # the optimizer is initialized.
        train_model.set_train_mode()
        optimizer, scheduler = train_model.get_optimizer(total_steps=config.steps)
        train_model, optimizer = fabric.setup(train_model, optimizer)  # type: ignore[assignment]

        logger.info(
            f"Resolved Args: {helpers.pretty_format_args(args=config.model_dump())}"
        )

        hyperparams = helpers.pretty_format_args_dict(config.model_dump())
        for logger_instance in fabric.loggers:
            logger_instance.log_hyperparams(hyperparams)

        state = TrainTaskState(
            train_model=train_model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            step=-1,
            model_class_path=train_model.get_task_model().class_path,
            model_init_args=train_model.get_task_model().init_args,
            # TODO(Guarin, 07/25): Add config to state. For this we have to make the config
            # JSON serializable.
        )

        if config.resume_interrupted:
            helpers.load_checkpoint(fabric=fabric, out_dir=out_dir, state=state)

        # TODO(Guarin, 07/25): Replace with infinite batch sampler instead to avoid
        # reloading dataloader after every epoch? Is this preferred over persistent workers?
        infinite_train_dataloader = InfiniteCycleIterator(iterable=train_dataloader)

        for name, param in train_model.named_parameters():
            logger.debug(f"grad={param.requires_grad} {name}")
        for name, module in train_model.named_modules():
            logger.debug(f"train={module.training} {name}")

        start_step = state["step"] + 1
        if start_step > 0:
            logger.info(f"Resuming training from step {start_step}/{config.steps}...")
        else:
            logger.info(f"Training for {config.steps} steps...")

        fabric.barrier()
        for step in range(start_step, config.steps):
            state["step"] = step
            is_last_step = step + 1 == config.steps
            is_log_step = (
                step == 0
                or (step + 1) % no_auto(config.logger_args.log_every_num_steps) == 0
            )
            is_val_step = (step + 1) % no_auto(
                config.logger_args.val_every_num_steps
            ) == 0
            is_save_ckpt_step = (step + 1) % no_auto(
                config.save_checkpoint_args.save_every_num_steps
            ) == 0

            batch = next(infinite_train_dataloader)
            train_result = train_model.training_step(
                fabric=fabric, batch=batch, step=step
            )
            fabric.backward(train_result.loss)
            train_model.clip_gradients(fabric=fabric, optimizer=optimizer)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if is_log_step or is_last_step:
                train_log_dict = helpers.compute_metrics(train_result.log_dict)
                helpers.log_step(
                    split="train",
                    step=step,
                    max_steps=config.steps,
                    log_dict=train_log_dict,
                )
                for group in optimizer.param_groups:
                    train_log_dict[f"learning_rate/{group['name']}"] = group["lr"]
                    train_log_dict[f"weight_decay/{group['name']}"] = group[
                        "weight_decay"
                    ]
                fabric.log_dict(train_log_dict, step=step)
                helpers.reset_metrics(train_result.log_dict)

            if is_save_ckpt_step or is_last_step:
                helpers.save_checkpoint(fabric=fabric, out_dir=out_dir, state=state)

            if is_val_step or is_last_step:
                fabric.barrier()
                logger.info("Validating...")
                train_model.eval()
                for val_step, val_batch in enumerate(val_dataloader):
                    is_last_val_step = val_step + 1 == len(val_dataloader)
                    is_val_log_step = val_step == 0 or (
                        (val_step + 1)
                        % no_auto(config.logger_args.val_log_every_num_steps)
                        == 0
                    )
                    with torch.no_grad():
                        val_result = train_model.validation_step(
                            fabric=fabric, batch=val_batch
                        )
                    if is_last_val_step:
                        val_log_dict = helpers.compute_metrics(val_result.log_dict)
                        helpers.log_step(
                            split="val",
                            step=val_step,
                            max_steps=len(val_dataloader),
                            log_dict=val_log_dict,
                        )
                        fabric.log_dict(val_log_dict, step=step)
                        helpers.reset_metrics(val_result.log_dict)
                    elif is_val_log_step:
                        # Show that we are making progress. Metrics are only calculated
                        # at the end of the validation loop.
                        helpers.log_step(
                            split="val",
                            step=val_step,
                            max_steps=len(val_dataloader),
                            log_dict={},
                        )
                train_model.set_train_mode()
                fabric.barrier()
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
    model_args: dict[str, Any] | TrainModelArgs | None = None
    transform_args: dict[str, Any] | None = None
    loader_args: dict[str, Any] | None = None
    save_checkpoint_args: dict[str, Any] | TaskSaveCheckpointArgs | None = None

    # Allow arbitrary field types such as Module, Dataset, Accelerator, ...
    model_config = ConfigDict(arbitrary_types_allowed=True)
