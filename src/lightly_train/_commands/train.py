#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

# Import old types for compatibility with omegaconf.
from typing import Any, Literal

import pytorch_lightning
from omegaconf import DictConfig
from pydantic import ConfigDict
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import (  # type: ignore[attr-defined]
    _PRECISION_INPUT,
)
from torch.nn import Module

from lightly_train import _logging, _system
from lightly_train._callbacks import callback_helpers
from lightly_train._callbacks.callback_args import CallbackArgs
from lightly_train._commands import _warnings, common_helpers, train_helpers
from lightly_train._commands.common_helpers import ModelFormat
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._loggers import logger_helpers
from lightly_train._loggers.logger_args import LoggerArgs
from lightly_train._methods import method_helpers
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models import package_helpers
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._transforms.transform import MethodTransformArgs
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


def train(
    out: PathLike,
    data: PathLike,
    model: str | Module,
    method: str = "distillation",
    method_args: dict[str, Any] | None = None,
    embed_dim: int | None = None,
    epochs: int = 100,
    batch_size: int = 128,
    num_workers: int | Literal["auto"] = "auto",
    devices: int | str | list[int] = "auto",
    num_nodes: int = 1,
    resume: bool = False,
    checkpoint: PathLike | None = None,
    overwrite: bool = False,
    accelerator: str | Accelerator = "auto",
    strategy: str | Strategy = "auto",
    precision: _PRECISION_INPUT = "32-true",  # Default precision in PyTorch Lightning
    seed: int = 0,
    loggers: dict[str, dict[str, Any] | None] | None = None,
    callbacks: dict[str, dict[str, Any] | None] | None = None,
    optim: str = "auto",
    optim_args: dict[str, Any] | None = None,
    transform_args: dict[str, Any] | None = None,
    loader_args: dict[str, Any] | None = None,
    trainer_args: dict[str, Any] | None = None,
    model_args: dict[str, Any] | None = None,
) -> None:
    """Train a self-supervised model.

    See the documentation for more information: https://docs.lightly.ai/train/stable/train.html

    The training process can be monitored with TensorBoard:

    .. code-block:: bash

        tensorboard --logdir out

    After training, the model is exported in the library default format to
    ``out/exported_models/exported_last.pt``. It can be exported to different formats
    using the ``lightly_train.export`` command.

    Args:
        out:
            Output directory to save logs, checkpoints, and other artifacts.
        data:
            Path to a directory containing images.
        model:
            Model name or instance to use for training.
        method:
            Self-supervised learning method name.
        method_args:
            Arguments for the self-supervised learning method. The available arguments
            depend on the ``method`` parameter.
        embed_dim:
            Embedding dimension. Set this if you want to train an embedding model with
            a specific dimension. If None, the output dimension of ``model`` is used.
        epochs:
            Number of training epochs.
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
        checkpoint:
            Checkpoint to load the model weights from. The checkpoint must be a file
            created by a previous training run. Apart from the weights, all other
            training state components (e.g. optimizer, epochs) are not loaded.
        resume:
            Resume training from the last checkpoint.
        overwrite:
            Overwrite the output directory if it already exists. Warning, this might
            overwrite existing files in the directory!
        accelerator:
            Hardware accelerator. Can be one of ['cpu', 'gpu', 'tpu', 'ipu', 'hpu',
            'mps', 'auto']. 'auto' will automatically select the best accelerator
            available.
        strategy:
            Training strategy. For example 'ddp' or 'auto'. 'auto' automatically
            selects the best strategy available.
        precision:
            Training precision. Select '16-mixed' for mixed 16-bit precision, '32-true'
            for full 32-bit precision, or 'bf16-mixed' for mixed bfloat16 precision.
        seed:
            Random seed for reproducibility.
        loggers:
            Loggers for training. Either None or a dictionary of logger names to either
            None or a dictionary of logger arguments. None uses the default loggers.
            To disable a logger, set it to None: ``loggers={"tensorboard": None}``.
            To configure a logger, pass the respective arguments:
            ``loggers={"wandb": {"project": "my_project"}}``.
        callbacks:
            Callbacks for training. Either None or a dictionary of callback names to
            either None or a dictionary of callback arguments. None uses the default
            callbacks. To disable a callback, set it to None:
            ``callbacks={"model_checkpoint": None}``. To configure a callback, pass the
            respective arguments:
            ``callbacks={"model_checkpoint": {"every_n_epochs": 5}}``.
        optim:
            Optimizer name. Must be one of ['auto', 'adamw', 'sgd']. 'auto' automatically
            selects the optimizer based on the method.
        optim_args:
            Optimizer arguments. Available arguments depend on the optimizer.

            AdamW:
                ``optim_args={"lr": float, "betas": (float, float), "weight_decay": float}``

            SGD:
                ``optim_args={"lr": float, "momentum": float, "weight_decay": float}``

        transform_args:
            Arguments for the image transform. The available arguments depend on the
            `method` parameter. The following arguments are always available:

            .. code-block:: python

                transform_args={
                    "image_size": (int, int),
                    "random_resize": {
                        "min_scale": float,
                        "max_scale": float,
                    },
                    "random_flip": {
                        "horizonal_prob": float,
                        "vertical_prob": float,
                    },
                    "random_rotation": {
                        "prob": float,
                        "degrees": int,
                    },
                    "random_gray_scale": float,
                    "normalize": {
                        "mean": (float, float, float),
                        "std": (float, float, float),
                    }
                }
        loader_args:
            Arguments for the PyTorch DataLoader. Should only be used in special cases
            as default values are automatically set. Prefer to use the `batch_size` and
            `num_workers` arguments instead. For details, see:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        trainer_args:
            Arguments for the PyTorch Lightning Trainer. Should only be used in special
            cases as default values are automatically set. For details, see:
            https://lightning.ai/docs/pytorch/stable/common/trainer.html
        model_args:
            Arguments for the model. The available arguments depend on the ``model``
            parameter. For example, if ``model='torchvision/<model_name>'``, the
            arguments are passed to
            ``torchvision.models.get_model(model_name, **model_args)``.
    """
    config = validate.pydantic_model_validate(TrainConfig, locals())
    train_from_config(config=config)


def train_from_config(config: TrainConfig) -> None:
    # Convert the config to a TrainConfig instance.
    config = validate.pydantic_model_validate(TrainConfig, dict(config))

    # Set up output directory.
    out_dir = common_helpers.get_out_dir(
        out=config.out, resume=config.resume, overwrite=config.overwrite
    )

    # Set up logging.
    _warnings.filter_train_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_file_logging(out_dir / "train.log")
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")
    logger.info(f"Using output directory '{out_dir}'.")

    # Log system information.
    system_information = _system.get_system_information()
    _system.log_system_information(system_information=system_information)

    pytorch_lightning.seed_everything(seed=config.seed, workers=True)
    config.transform_args = train_helpers.get_transform_args(
        method=config.method, transform_args=config.transform_args
    )
    transform_instance = train_helpers.get_transform(
        method=config.method, transform_args_resolved=config.transform_args
    )
    # Create a temporary file to use as a memory map for dataset items. The
    # file has to exist while the dataset is used.
    # TODO(Philipp, 10/24): For training it could make sense to store the
    # file in the output directory and recover it on resume.
    with common_helpers.get_dataset_temp_mmap_path() as mmap_filepath:
        dataset = common_helpers.get_dataset(
            data=config.data,
            transform=transform_instance,
            mmap_filepath=mmap_filepath,
        )
        scaling_info = train_helpers.get_scaling_info(
            dataset=dataset, epochs=config.epochs
        )
        model_instance = package_helpers.get_model(
            model=config.model, model_args=config.model_args
        )
        embedding_model = train_helpers.get_embedding_model(
            model=model_instance, embed_dim=config.embed_dim
        )
        log_every_n_steps = train_helpers.get_lightning_logging_interval(
            dataset_size=scaling_info.dataset_size, batch_size=config.batch_size
        )
        config.loggers = logger_helpers.get_logger_args(loggers=config.loggers)
        logger_instances = logger_helpers.get_loggers(
            logger_args=config.loggers, out=out_dir
        )
        config.callbacks = callback_helpers.get_callback_args(
            callback_args=config.callbacks
        )
        callback_instances = callback_helpers.get_callbacks(
            callback_args=config.callbacks,
            out=out_dir,
            model=callback_helpers.get_checkpoint_model(model=model_instance),
            embedding_model=embedding_model,
            normalize_args=transform_instance.transform_args.normalize,
        )
        config.accelerator = common_helpers.get_accelerator(
            accelerator=config.accelerator
        )
        config.strategy = train_helpers.get_strategy(
            accelerator=config.accelerator,
            strategy=config.strategy,
            devices=config.devices,
        )
        trainer_instance = train_helpers.get_trainer(
            out=out_dir,
            epochs=config.epochs,
            accelerator=config.accelerator,
            strategy=config.strategy,
            devices=config.devices,
            num_nodes=config.num_nodes,
            precision=config.precision,
            log_every_n_steps=log_every_n_steps,
            loggers=logger_instances,
            callbacks=callback_instances,
            trainer_args=config.trainer_args,
        )
        config.accelerator = trainer_instance.accelerator
        config.strategy = trainer_instance.strategy
        config.devices = trainer_instance.num_devices

        total_num_devices = train_helpers.get_total_num_devices(
            num_nodes=trainer_instance.num_nodes,
            num_devices=trainer_instance.num_devices,
        )
        config.batch_size = train_helpers.get_global_batch_size(
            global_batch_size=config.batch_size,
            dataset=dataset,
            total_num_devices=total_num_devices,
            loader_args=config.loader_args,
        )
        config.num_workers = common_helpers.get_num_workers(
            num_workers=config.num_workers,
            num_devices_per_node=total_num_devices // trainer_instance.num_nodes,
        )
        dataloader = train_helpers.get_dataloader(
            dataset=dataset,
            batch_size=config.batch_size // total_num_devices,
            num_workers=config.num_workers,
            loader_args=config.loader_args,
        )
        method_cls = method_helpers.get_method_cls(method=config.method)
        config.optim_args = train_helpers.get_optimizer_args(
            optim_type=train_helpers.get_optimizer_type(optim_type=config.optim),
            optim_args=config.optim_args,
            method_cls=method_cls,
        )
        config.optim = config.optim_args.type().value
        config.method_args = train_helpers.get_method_args(
            method_cls=method_cls,
            method_args=config.method_args,
            scaling_info=scaling_info,
            optimizer_args=config.optim_args,
        )
        method_instance = train_helpers.get_method(
            method_cls=method_cls,
            method_args=config.method_args,
            optimizer_args=config.optim_args,
            embedding_model=embedding_model,
            global_batch_size=config.batch_size,
        )
        train_helpers.load_checkpoint(
            checkpoint=config.checkpoint,
            resume=config.resume,
            model=model_instance,
            embedding_model=embedding_model,
            method=method_instance,
        )
        log_resolved_config(config=config, loggers=logger_instances)
        trainer_instance.fit(
            model=method_instance,
            train_dataloaders=dataloader,
            ckpt_path="last" if config.resume else None,
        )
    if config.epochs == 0:
        logger.info("No training epochs specified. Saving model and exiting.")
        trainer_instance.save_checkpoint(out_dir / "checkpoints" / "last.ckpt")
    logger.info("Training completed.")
    common_helpers.export_model(
        model=model_instance,
        out=out_dir / "exported_models" / "exported_last.pt",
        format=ModelFormat.PACKAGE_DEFAULT,
    )
    logger.info("Model exported.")


def train_from_dictconfig(config: DictConfig) -> None:
    logger.debug(f"Training model with config: {config}")
    config_dict = omegaconf_utils.config_to_dict(config=config)
    train_cfg = validate.pydantic_model_validate(CLITrainConfig, config_dict)
    train_from_config(config=train_cfg)


class TrainConfig(PydanticConfig):
    out: PathLike
    data: PathLike
    model: str | Module
    method: str = "distillation"
    method_args: dict[str, Any] | MethodArgs | None = None
    embed_dim: int | None = None
    epochs: int = 100
    batch_size: int = 128
    num_workers: int | Literal["auto"] = "auto"
    devices: int | str | list[int] = "auto"
    num_nodes: int = 1
    resume: bool = False
    checkpoint: PathLike | None = None
    overwrite: bool = False
    accelerator: str | Accelerator = "auto"
    strategy: str | Strategy = "auto"
    precision: _PRECISION_INPUT = "32-true"
    seed: int = 0
    loggers: dict[str, dict[str, Any] | None] | LoggerArgs | None = None
    callbacks: dict[str, dict[str, Any] | None] | CallbackArgs | None = None
    optim: str | OptimizerType = "auto"
    optim_args: dict[str, Any] | OptimizerArgs | None = None
    transform_args: dict[str, Any] | MethodTransformArgs | None = None
    loader_args: dict[str, Any] | None = None
    trainer_args: dict[str, Any] | None = None
    model_args: dict[str, Any] | None = None

    # Allow arbitrary field types such as Module, Dataset, Accelerator, ...
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FunctionTrainConfig(TrainConfig):
    # Configuration with simpler types for calling the train function.
    method_args: dict[str, Any] | None = None
    loggers: dict[str, dict[str, Any] | None] | None = None
    callbacks: dict[str, dict[str, Any] | None] | None = None
    optim: str = "auto"
    optim_args: dict[str, Any] | None = None
    transform_args: dict[str, Any] | None = None


class CLITrainConfig(FunctionTrainConfig):
    # CLI configuration with simpler types for better error messages.
    out: str
    data: str
    model: str
    checkpoint: str | None = None
    accelerator: str = "auto"
    strategy: str = "auto"

    # CLI should not pass arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=False)


def log_resolved_config(config: TrainConfig, loggers: list[Logger]) -> None:
    """Log the resolved configuration.

    Note that the resolved configuration might still have a few values set to "auto":
    - config.strategy
    - config.devices
    """
    log_string = (
        "Resolved configuration:\n"
        f"{common_helpers.pretty_format_args(args=config.model_dump())}"
    )
    logger.info(log_string)

    hyperparams = common_helpers.sanitize_config_dict(config.model_dump())
    for logger_instance in loggers:
        logger_instance.log_hyperparams(params=hyperparams)
