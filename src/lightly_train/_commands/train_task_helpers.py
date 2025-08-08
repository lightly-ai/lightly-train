#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import logging
from functools import partial
from json import JSONEncoder
from pathlib import Path
from typing import Any, Literal

import torch
from lightning_fabric import Fabric
from lightning_fabric import utilities as fabric_utilities
from lightning_fabric.loggers.logger import Logger as FabricLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lightly_train._configs import validate
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train._env import Env
from lightly_train._loggers.mlflow import MLFlowLogger
from lightly_train._loggers.task_logger_args import TaskLoggerArgs
from lightly_train._loggers.tensorboard import TensorBoardLogger
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.dinov2_eomt_semantic_segmentation.train_model import (
    DINOv2EoMTSemanticSegmentationTrain,
    DINOv2EoMTSemanticSegmentationTrainArgs,
)
from lightly_train._task_models.dinov2_eomt_semantic_segmentation.transforms import (
    DINOv2SemanticSegmentationTrainTransform,
    DINOv2SemanticSegmentationTrainTransformArgs,
    DINOv2SemanticSegmentationValTransform,
    DINOv2SemanticSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TrainModel,
    TrainModelArgs,
)
from lightly_train._train_task_state import TrainTaskState
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import (
    MaskSemanticSegmentationBatch,
    MaskSemanticSegmentationDatasetItem,
    PathLike,
    TaskDatasetItem,
)

logger = logging.getLogger(__name__)


def get_out_dir(
    fabric: Fabric,
    out: PathLike,
    resume_interrupted: bool,
    overwrite: bool,
) -> Path:
    # Use the same output directory on all ranks. This avoids issues where users
    # accidentally create different directories on each rank, for example with:
    #   out=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_global_rank_zero = fabric.broadcast(str(out))
    out_dir = Path(out_global_rank_zero)

    def check_and_create_out_dir() -> None:
        if out_dir.exists():
            if not out_dir.is_dir():
                raise ValueError(f"Output '{out_dir}' is not a directory!")

            dir_not_empty = any(out_dir.iterdir())

            if dir_not_empty and (not (resume_interrupted or overwrite)):
                raise ValueError(
                    f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite "
                    "the directory or resume_interrupted=True to resume training from "
                    "an interrupted or crashed run. "
                    "See https://docs.lightly.ai/lightly-train/usage/cli.html#resume-training "
                    "for more information on how to resume training."
                )
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

    # Create the output directory if it doesn't exist.
    with fabric.rank_zero_first():
        if fabric.global_rank == 0:
            check_and_create_out_dir()

    # Check if the output directory is on a shared filesystem. We can only check this
    # after global rank zero has created the directory.
    try:
        is_shared_filesystem = fabric_utilities.is_shared_filesystem(
            strategy=fabric.strategy, path=out_dir
        )
    except FileNotFoundError:
        # Clearly not a shared filesystem because we just created the directory.
        is_shared_filesystem = False

    # If the filesystem is not shared we have to create the output directory on every
    # node individually.
    if not is_shared_filesystem:
        with fabric.rank_zero_first(local=True):
            if fabric.local_rank == 0 and fabric.global_rank != 0:
                check_and_create_out_dir()

    return out_dir


def get_logger_args(
    steps: int,
    val_steps: int,
    logger_args: dict[str, Any] | TaskLoggerArgs | None = None,
) -> TaskLoggerArgs:
    if isinstance(logger_args, TaskLoggerArgs):
        return logger_args
    logger_args = {} if logger_args is None else logger_args
    args = validate.pydantic_model_validate(TaskLoggerArgs, logger_args)
    args.resolve_auto(steps=steps, val_steps=val_steps)
    return args


def get_loggers(logger_args: TaskLoggerArgs, out: Path) -> list[FabricLogger]:
    """Get logger instances based on the provided configuration.

    All loggers are configured with the same output directory 'out'.

    Args:
        logger_args:
            Configuration for the loggers.
        out:
            Path to the output directory.

    Returns:
        List of loggers.
    """
    loggers: list[FabricLogger] = []

    if logger_args.mlflow is not None:
        logger.debug(f"Using mlflow logger with args {logger_args.mlflow}")
        loggers.append(MLFlowLogger(save_dir=out, **logger_args.mlflow.model_dump()))
    if logger_args.tensorboard is not None:
        logger.debug(f"Using tensorboard logger with args {logger_args.tensorboard}")
        loggers.append(
            TensorBoardLogger(save_dir=out, **logger_args.tensorboard.model_dump())
        )

    logger.debug(f"Using loggers {[log.__class__.__name__ for log in loggers]}.")
    return loggers


class PrettyFormatArgsJSONEncoder(JSONEncoder):
    """Custom JSON encoder to pretty format the output."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            # Return class name for objects that cannot be serialized
            return obj.__class__.__name__


def pretty_format_args(args: dict[str, Any], indent: int = 4) -> str:
    return json.dumps(
        args, indent=indent, sort_keys=True, cls=PrettyFormatArgsJSONEncoder
    )


def pretty_format_args_dict(args: dict[str, Any]) -> dict[str, Any]:
    args_str = json.dumps(args, cls=PrettyFormatArgsJSONEncoder)
    args_dict: dict[str, Any] = json.loads(args_str)
    return args_dict


def get_train_transform(ignore_index: int) -> TaskTransform:
    return DINOv2SemanticSegmentationTrainTransform(
        DINOv2SemanticSegmentationTrainTransformArgs(
            ignore_index=ignore_index,
        )
    )


def get_val_transform(ignore_index: int) -> TaskTransform:
    return DINOv2SemanticSegmentationValTransform(
        DINOv2SemanticSegmentationValTransformArgs(
            ignore_index=ignore_index,
        )
    )


def get_dataset(
    dataset_args: MaskSemanticSegmentationDatasetArgs, transform: TaskTransform
) -> MaskSemanticSegmentationDataset:
    # TODO(Guarin, 07/25): MMAP filenames.
    filenames = list(dataset_args.list_image_filenames())
    dataset_cls = dataset_args.get_dataset_cls()
    return dataset_cls(
        dataset_args=dataset_args, image_filenames=filenames, transform=transform
    )

# TODO(Guarin, 08/25): Move this to function to the _data module.
def collate_fn(
    batch: list[MaskSemanticSegmentationDatasetItem], split: str
) -> MaskSemanticSegmentationBatch:
    # Prepare the batch without any stacking.
    out: MaskSemanticSegmentationBatch = {
        "image_path": [item["image_path"] for item in batch],
        "image": [item["image"] for item in batch],
        "mask": [item["mask"] for item in batch],
        "binary_masks": [item["binary_masks"] for item in batch],
    }

    # During training images and masks all have the same shape.
    if split == "train":
        out["image"] = torch.stack(out["image"])
        out["mask"] = torch.stack(out["mask"])

    return out


def get_train_dataloader(
    fabric: Fabric,
    dataset: Dataset[TaskDatasetItem],
    batch_size: int,
    num_workers: int,
    loader_args: dict[str, Any] | None = None,
) -> DataLoader[TaskDatasetItem]:
    timeout = Env.LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0
    # TODO(Guarin, 07/25): Persistent workers by default?
    dataloader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size // fabric.world_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        timeout=timeout,
        collate_fn=partial(collate_fn, split="train"),
    )
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        dataloader_kwargs.update(**loader_args)
    dataloader = DataLoader(**dataloader_kwargs)
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value,no-any-return]


def get_val_dataloader(
    fabric: Fabric,
    dataset: Dataset[TaskDatasetItem],
    batch_size: int,
    num_workers: int,
    loader_args: dict[str, Any] | None = None,
) -> DataLoader[TaskDatasetItem]:
    timeout = Env.LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0
    dataloader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=batch_size // fabric.world_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        timeout=timeout,
        collate_fn=partial(collate_fn, split="validation"),
    )
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        dataloader_kwargs.update(**loader_args)
    dataloader = DataLoader(**dataloader_kwargs)
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value,no-any-return]


def get_steps(steps: int | Literal["auto"], default_steps: int) -> int:
    return default_steps if steps == "auto" else steps


def get_train_model_args_cls(
    model_name: str, model_args: dict[str, Any] | TrainModelArgs | None
) -> type[TrainModelArgs]:
    if isinstance(model_args, TrainModelArgs):
        return model_args.__class__

    # TODO(Guarin, 08/25): Properly handle model name and args linking.
    if model_name.endswith("-eomt"):
        return DINOv2EoMTSemanticSegmentationTrainArgs
    raise ValueError(f"Unsupported model name '{model_name}'.")


def get_train_model_args(
    model_args: dict[str, Any] | TrainModelArgs | None,
    model_args_cls: type[TrainModelArgs],
    total_steps: int,
) -> TrainModelArgs:
    if isinstance(model_args, TrainModelArgs):
        return model_args
    model_args = {} if model_args is None else model_args
    args = validate.pydantic_model_validate(model_args_cls, model_args)
    args.resolve_auto(total_steps=total_steps)
    return args


def get_train_model(
    model_name: str,
    model_args: TrainModelArgs,
    data_args: MaskSemanticSegmentationDataArgs,
) -> TrainModel:
    package, model = model_name.split("/", maxsplit=1)
    if package == "dinov2_vit":  # For backwards compatibility
        package = "dinov2"
    if package != "dinov2":
        raise ValueError(
            f"Unsupported model '{model_name}'. Only 'dinov2' models are supported."
        )
    assert isinstance(model_args, DINOv2EoMTSemanticSegmentationTrainArgs)
    return DINOv2EoMTSemanticSegmentationTrain(
        model_args=model_args, model_name=model, data_args=data_args
    )


def log_step(
    split: Literal["train", "val"], step: int, max_steps: int, log_dict: dict[str, Any]
) -> None:
    split_cap = split.capitalize()
    name_to_display_name = {
        "train_loss": "Train Loss",
        "train_metric/miou": "Train mIoU",
        "val_loss": "Val Loss",
        "val_metric/miou": "Val mIoU",
    }
    parts = [
        f"{split_cap} Step {step + 1}/{max_steps}",
    ]
    for name, value in log_dict.items():
        if name in name_to_display_name:
            parts.append(f"{name_to_display_name[name]}: {value:.4f}")
    line = " | ".join(parts)
    logger.info(line)


def compute_metrics(log_dict: dict[str, Any]) -> dict[str, Any]:
    # Lazy import because torchmetrics is optional dependency.
    from torchmetrics import Metric

    metrics = {}
    for name, value in log_dict.items():
        if isinstance(value, Metric):
            value = value.compute()
        if isinstance(value, Tensor) and value.numel() > 1:
            for i, v in enumerate(value):
                metrics[f"{name}_{i}"] = v.item()
        if isinstance(value, dict):
            for class_name, class_value in value.items():
                metrics[f"{name}{class_name}"] = class_value.item()
        else:
            metrics[name] = value
    return metrics


def reset_metrics(log_dict: dict[str, Any]) -> None:
    # Lazy import because torchmetrics is optional dependency.
    from torchmetrics import Metric

    for value in log_dict.values():
        if isinstance(value, Metric):
            value.reset()


def get_save_checkpoint_args(
    checkpoint_args: dict[str, Any] | TaskSaveCheckpointArgs | None,
) -> TaskSaveCheckpointArgs:
    if isinstance(checkpoint_args, TaskSaveCheckpointArgs):
        return checkpoint_args
    checkpoint_args = {} if checkpoint_args is None else checkpoint_args
    args = validate.pydantic_model_validate(TaskSaveCheckpointArgs, checkpoint_args)
    return args


def get_last_checkpoint_path(out_dir: PathLike) -> Path:
    out_dir = Path(out_dir).resolve()
    ckpt_path = out_dir / "checkpoints" / "last.ckpt"
    return ckpt_path


def save_checkpoint(fabric: Fabric, out_dir: Path, state: TrainTaskState) -> None:
    ckpt_path = get_last_checkpoint_path(out_dir)
    logger.info(f"Saving checkpoint to '{ckpt_path}'")
    fabric.save(path=ckpt_path, state=state)  # type: ignore[arg-type]


def load_checkpoint(fabric: Fabric, out_dir: PathLike, state: TrainTaskState) -> None:
    ckpt_path = get_last_checkpoint_path(out_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path}' does not exist.")

    train_model = state["train_model"]
    train_model_grads = {n: p.requires_grad for n, p in train_model.named_parameters()}
    train_model_trainings = {n: m.training for n, m in train_model.named_modules()}
    optimizer = state["optimizer"]
    train_dataloader = state["train_dataloader"]

    logger.info(f"Loading checkpoint from '{ckpt_path}'")
    fabric.load(path=ckpt_path, state=state)  # type: ignore[arg-type]

    # Sanity check to make sure that checkpoint loading didn't create new objects or
    # changed the model state.
    assert state["train_model"] is train_model
    assert {
        n: p.requires_grad for n, p in state["train_model"].named_parameters()
    } == train_model_grads
    assert {
        n: m.training for n, m in state["train_model"].named_modules()
    } == train_model_trainings
    assert state["optimizer"] is optimizer
    assert state["train_dataloader"] is train_dataloader
