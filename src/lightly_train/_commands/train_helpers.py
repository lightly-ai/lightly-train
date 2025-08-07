#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal, Sized

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.loggers import Logger
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import (  # type: ignore[attr-defined]
    _PRECISION_INPUT,
)
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from lightly_train._checkpoint import Checkpoint
from lightly_train._configs import validate
from lightly_train._env import Env
from lightly_train._methods import method_helpers
from lightly_train._methods.method import Method
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._optim import optimizer_helpers
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo
from lightly_train._transforms.transform import (
    MethodTransform,
    MethodTransformArgs,
)
from lightly_train.types import DatasetItem, PathLike

logger = logging.getLogger(__name__)


def get_transform_args(
    method: str | Method, transform_args: dict[str, Any] | MethodTransformArgs | None
) -> MethodTransformArgs:
    logger.debug(f"Getting transform args for method '{method}'.")
    logger.debug(f"Using additional transform arguments {transform_args}.")
    if isinstance(transform_args, MethodTransformArgs):
        return transform_args

    method_cls = method_helpers.get_method_cls(method)
    transform_cls = method_cls.transform_cls()
    transform_args_cls = transform_cls.transform_args_cls()

    if transform_args is None:
        # We need to typeignore here because a MethodTransformArgs might not have
        # defaults for all fields, while its children do.
        return transform_args_cls()  # type: ignore[call-arg]

    return validate.pydantic_model_validate(transform_args_cls, transform_args)


def get_transform(
    method: str | Method,
    transform_args_resolved: MethodTransformArgs,
) -> MethodTransform:
    logger.debug(f"Getting transform for method '{method}'.")
    validate.assert_config_resolved(transform_args_resolved)
    method_cls = method_helpers.get_method_cls(method)
    transform_cls = method_cls.transform_cls()
    transform = transform_cls(transform_args=transform_args_resolved)
    return transform


def get_total_num_devices(
    num_nodes: int,
    num_devices: int,
) -> int:
    """Returns the total number of devices across all nodes.

    Assumes that all nodes have the same number of devices.
    """
    # NOTE(Guarin, 09/24): We don't use the trainer.world_size attribute to calculate
    # the total number of devices because it uses SLURM_NTASKS to determined the number
    # of devices which doesn't always match SLURM_NODES * SLURM_NTASKS_PER_NODE.
    total_num_devices = num_nodes * num_devices
    logger.debug(f"Detected {num_nodes} nodes and {num_devices} devices per node.")
    logger.debug(f"Total number of devices: {total_num_devices}.")
    return total_num_devices


def get_dataloader(
    dataset: Dataset[DatasetItem],
    batch_size: int,
    num_workers: int,
    loader_args: dict[str, Any] | None,
) -> DataLoader[DatasetItem]:
    """Creates a dataloader for the given dataset.

    Args:
        dataset:
            Dataset.
        batch_size:
            Batch size per dataloader. This is the batch size per device.
        num_workers:
            Number of workers for the dataloader.
        loader_args:
            Additional arguments for the DataLoader. Additional arguments have priority
            over other arguments.
    """
    logger.debug(f"Using batch size per device {batch_size}.")
    timeout = Env.LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0
    dataloader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        timeout=timeout,
    )
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        dataloader_kwargs.update(**loader_args)
    return DataLoader(**dataloader_kwargs)


def get_embedding_model(
    wrapped_model: ModelWrapper, embed_dim: int | None = None
) -> EmbeddingModel:
    logger.debug(f"Getting embedding model with embedding dimension {embed_dim}.")
    return EmbeddingModel(wrapped_model=wrapped_model, embed_dim=embed_dim)


def get_trainer(
    out: Path,
    epochs: int,
    accelerator: str | Accelerator,
    strategy: str | Strategy,
    devices: list[int] | str | int,
    num_nodes: int,
    log_every_n_steps: int,
    precision: _PRECISION_INPUT | None,
    loggers: list[Logger],
    callbacks: list[Callback],
    trainer_args: dict[str, Any] | None,
) -> Trainer:
    logger.debug("Getting trainer.")

    sync_batchnorm = get_sync_batchnorm(accelerator=accelerator)

    trainer_kwargs: dict[str, Any] = dict(
        default_root_dir=out,
        max_epochs=epochs,
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=loggers,
        sync_batchnorm=sync_batchnorm,
    )
    if trainer_args is not None:
        logger.debug(f"Using additional trainer arguments {trainer_args}.")
        trainer_kwargs.update(trainer_args)

    return Trainer(**trainer_kwargs)


def get_lightning_logging_interval(dataset_size: int, batch_size: int) -> int:
    """Calculates the logging interval for the given dataset and batch size.

    If the number of batches is smaller than the logging interval, Lightning
    raises a UserWarning. To avoid this, we take the minimum of 50 and the number
    of batches.
    """
    if dataset_size <= 0 or batch_size <= 0:
        raise ValueError(
            f"Dataset size ({dataset_size}) and batch size ({batch_size}) must be positive integers."
        )
    n_batches = max(1, dataset_size // batch_size)
    return min(50, n_batches)  # Lightning uses 50 as default logging interval.


def get_strategy(
    strategy: str | Strategy,
    accelerator: str | Accelerator,
    devices: list[int] | str | int,
) -> str | Strategy:
    if strategy != "auto":
        logger.debug(f"Using provided strategy '{strategy}'.")
        return strategy

    accelerator_cls: type[CUDAAccelerator] | type[CPUAccelerator]
    if isinstance(accelerator, CUDAAccelerator) or accelerator == "gpu":
        accelerator_cls = CUDAAccelerator
    elif isinstance(accelerator, CPUAccelerator) or accelerator == "cpu":
        accelerator_cls = CPUAccelerator
    else:
        # For non CPU/CUDA accelerators we let PyTorch Lightning decide.
        logger.debug(
            "Non CPU/CUDA accelerator, using default strategy by PyTorchLightning."
        )
        return strategy

    if devices == "auto":
        num_devices = accelerator_cls.auto_device_count()
    else:
        if isinstance(devices, list):
            # Workaround as CUDAAccelerator.parse_devices doesn't accept a list of ints.
            devices = ",".join(map(str, devices))

        parsed_devices = accelerator_cls.parse_devices(devices=devices)
        # None means that no devices were requested.
        if parsed_devices is None:
            return strategy
        num_devices = (
            len(parsed_devices) if isinstance(parsed_devices, list) else parsed_devices
        )
    logger.debug(f"Detected {num_devices} devices.")

    if num_devices > 1:
        # If we have multiple CPU or CUDA devices, use DDP with find_unused_parameters.
        # find_unused_parameters avoids DDP errors for models/methods that have
        # extra parameters which are not used in all the forward passes. This is for
        # example the case in DINO where the projection head is frozen during the first
        # epoch.
        # TODO: Only set find_unused_parameters=True if necessary as it slows down
        # training speed. See https://github.com/pytorch/pytorch/pull/44826 on how
        # parameters can be ignored for DDP.
        logger.debug("Using strategy 'ddp_find_unused_parameters_true'.")
        return "ddp_find_unused_parameters_true"

    logger.debug(f"Using strategy '{strategy}'.")
    return strategy


def get_sync_batchnorm(accelerator: str | Accelerator) -> bool:
    # SyncBatchNorm is only supported on CUDA devices.
    assert accelerator != "auto"
    use_sync_batchnorm = accelerator == "gpu" or isinstance(
        accelerator, CUDAAccelerator
    )
    logger.debug(f"Using sync_batchnorm '{use_sync_batchnorm}'.")
    return use_sync_batchnorm


def get_optimizer_type(
    optim_type: str | OptimizerType,
) -> OptimizerType | Literal["auto"]:
    # Auto is handled here and not in optimizer_helpers because the Method is in the end
    # responsible for automatically choosing the optimizer and we don't want to
    # introduce an "auto" optimizer type.
    if optim_type == "auto":
        return "auto"
    return optimizer_helpers.get_optimizer_type(optim_type=optim_type)


def get_optimizer_args(
    optim_type: OptimizerType | Literal["auto"],
    optim_args: dict[str, Any] | OptimizerArgs | None,
    method_cls: type[Method],
) -> OptimizerArgs:
    if isinstance(optim_args, OptimizerArgs):
        return optim_args
    optim_args = {} if optim_args is None else optim_args
    optim_args_cls = method_cls.optimizer_args_cls(optim_type=optim_type)
    logger.debug(f"Using optimizer '{optim_args_cls.type()}'.")
    return validate.pydantic_model_validate(optim_args_cls, optim_args)


def get_dataset_size(
    dataset: Dataset[DatasetItem],
) -> int:
    if isinstance(dataset, Sized):
        dataset_size = len(dataset)
        logger.debug(f"Found dataset size {dataset_size}.")
    else:
        logger.debug(
            f"Dataset does not have a length. Using default dataset size {IMAGENET_SIZE}."
        )
        dataset_size = IMAGENET_SIZE
    return dataset_size


def get_scaling_info(
    dataset_size: int,
    epochs: int,
) -> ScalingInfo:
    return ScalingInfo(dataset_size=dataset_size, epochs=epochs)


def get_method_args(
    method_cls: type[Method],
    method_args: dict[str, Any] | MethodArgs | None,
    scaling_info: ScalingInfo,
    optimizer_args: OptimizerArgs,
    wrapped_model: ModelWrapper,
) -> MethodArgs:
    logger.debug(f"Getting method args for '{method_cls.__name__}'")
    if isinstance(method_args, MethodArgs):
        return method_args
    method_args = {} if method_args is None else method_args
    method_args_cls = method_cls.method_args_cls()
    args = validate.pydantic_model_validate(method_args_cls, method_args)
    args.resolve_auto(
        scaling_info=scaling_info,
        optimizer_args=optimizer_args,
        wrapped_model=wrapped_model,
    )
    return args


def get_method(
    method_cls: type[Method],
    method_args: MethodArgs,
    optimizer_args: OptimizerArgs,
    embedding_model: EmbeddingModel,
    global_batch_size: int,
) -> Method:
    logger.debug(f"Getting method for '{method_cls.__name__}'")
    return method_cls(
        method_args=method_args,
        optimizer_args=optimizer_args,
        embedding_model=embedding_model,
        global_batch_size=global_batch_size,
    )


def get_epochs(
    method: str, epochs: int | Literal["auto"], dataset_size: int, batch_size: int
) -> int:
    method_args_cls = method_helpers.get_method_cls(method).method_args_cls()

    assert not (
        method_args_cls.default_epochs is None and method_args_cls.default_steps is None
    )
    assert not (
        isinstance(method_args_cls.default_epochs, int)
        and isinstance(method_args_cls.default_steps, int)
    )

    if epochs != "auto":
        logger.debug(f"Using provided epochs {epochs}.")
        return epochs

    if method_args_cls.default_epochs is not None:
        logger.debug(f"Using default epochs {method_args_cls.default_epochs}.")
        return method_args_cls.default_epochs
    elif method_args_cls.default_steps is not None:
        logger.debug(f"Using default steps {method_args_cls.default_steps}.")
        # Calculate epochs from steps.
        _epochs = math.ceil(method_args_cls.default_steps * batch_size / dataset_size)
        logger.debug(
            f"Calculated epochs {epochs} from steps {method_args_cls.default_steps}."
        )
        return _epochs
    else:
        raise ValueError(
            f"An unexpected error occurred while determining the number of epochs for method '{method_args_cls.__name__}'. "
            "Please contact the Lightly team."
        )


def load_checkpoint(
    checkpoint: PathLike | None,
    resume_interrupted: bool,
    wrapped_model: ModelWrapper,
    embedding_model: EmbeddingModel,
    method: Method,
) -> None:
    if checkpoint is not None:
        if resume_interrupted:
            raise ValueError(
                f"resume_interrupted={resume_interrupted} and checkpoint='{checkpoint}' "
                "cannot be set at the same time! Please set only one of them. "
                "See https://docs.lightly.ai/train/stable/train/index.html#resume-training "
                "for more information on which option to use."
            )
        logger.info(f"Loading model weights from '{checkpoint}'.")
        load_state_dict(
            wrapped_model=wrapped_model,
            embedding_model=embedding_model,
            method=method,
            checkpoint=checkpoint,
        )


def load_state_dict(
    wrapped_model: ModelWrapper,
    embedding_model: EmbeddingModel,
    method: Method,
    checkpoint: PathLike,
) -> None:
    ckpt = Checkpoint.from_path(Path(checkpoint))
    wrapped_model.load_state_dict(ckpt.lightly_train.models.wrapped_model.state_dict())
    model = wrapped_model.get_model()
    if isinstance(model, Module):
        model.load_state_dict(ckpt.lightly_train.models.model.state_dict())
    embedding_model.load_state_dict(
        ckpt.lightly_train.models.embedding_model.state_dict()
    )
    method.load_state_dict(ckpt.state_dict)
