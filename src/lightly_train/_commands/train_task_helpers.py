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
from contextlib import contextmanager
from json import JSONEncoder
from pathlib import Path
from typing import Any, Generator, Literal

from lightning_fabric import Fabric
from lightning_fabric import utilities as fabric_utilities
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric

from lightly_train._configs import validate
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train._env import Env
from lightly_train._loggers.task_logger_args import TaskLoggerArgs
from lightly_train._task_checkpoint import TaskCheckpointArgs
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation_train import (
    DINOv2SemanticSegmentationTrain,
    DINOv2SemanticSegmentationTrainArgs,
)
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation_transforms import (
    DINOv2SemanticSegmentationTrainTransform,
    DINOv2SemanticSegmentationTrainTransformArgs,
    DINOv2SemanticSegmentationValTransform,
    DINOv2SemanticSegmentationValTransformArgs,
)
from lightly_train._task_models.task_train_model import (
    TaskTrainModel,
    TaskTrainModelArgs,
)
from lightly_train._train_task_state import TrainTaskState
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import PathLike, TaskDatasetItem

logger = logging.getLogger(__name__)


@contextmanager
def filesystem_rank_zero_first(
    fabric: Fabric, path: PathLike
) -> Generator[bool, None, None]:
    """The code under this context manager is first excecuted on rank zero for the
    filesystem at `path`.

    If the filesystem at path is shared, the code is first executed on global rank zero.
    If the filesystem at path is not shared, the code is first executed on all local
    zero ranks.

    Yields:
        Boolean that indicates whether the code in the current process is executed
        first.
    """
    is_shared = fabric_utilities.is_shared_filesystem(
        strategy=fabric.strategy, path=path
    )
    local = not is_shared
    is_filesystem_rank_zero = fabric.is_global_zero or (
        local and fabric.local_rank == 0
    )
    with fabric.rank_zero_first(local=local):
        yield is_filesystem_rank_zero


def get_out_dir(
    fabric: Fabric, out: PathLike, resume_interrupted: bool, overwrite: bool
) -> Path:
    out_dir = Path(out).resolve()
    with filesystem_rank_zero_first(
        fabric=fabric, path=out_dir
    ) as is_filesystem_rank_zero:
        if not is_filesystem_rank_zero:
            return out_dir

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


def get_train_transform() -> TaskTransform:
    return DINOv2SemanticSegmentationTrainTransform(
        DINOv2SemanticSegmentationTrainTransformArgs()
    )


def get_val_transform() -> TaskTransform:
    return DINOv2SemanticSegmentationValTransform(
        DINOv2SemanticSegmentationValTransformArgs()
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        timeout=timeout,
    )
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        dataloader_kwargs.update(**loader_args)
    dataloader = DataLoader(**dataloader_kwargs)
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value,no-any-return]


def get_steps(steps: int | Literal["auto"]) -> int:
    if steps == "auto":
        return 1000
    return steps


def get_batch_size(batch_size: int | Literal["auto"]) -> int:
    if batch_size == "auto":
        return 32
    return batch_size


def get_task_train_model_args(
    task_args: dict[str, Any] | TaskTrainModelArgs | None,
) -> TaskTrainModelArgs:
    if isinstance(task_args, TaskTrainModelArgs):
        return task_args
    task_args = {} if task_args is None else task_args
    task_cls = DINOv2SemanticSegmentationTrainArgs
    args = validate.pydantic_model_validate(task_cls, task_args)
    return args


def get_task_train_model(
    model_name: str,
    task_args: TaskTrainModelArgs,
    data_args: MaskSemanticSegmentationDataArgs,
) -> TaskTrainModel:
    package, model = model_name.split("/", maxsplit=1)
    if package != "dinov2_vit":
        raise ValueError(
            f"Unsupported model '{model_name}'. Only 'dinov2_vit' models are supported."
        )
    assert isinstance(task_args, DINOv2SemanticSegmentationTrainArgs)
    return DINOv2SemanticSegmentationTrain(
        task_args=task_args, model_name=model, data_args=data_args
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
        parts.append(f"{name_to_display_name[name]}: {value:.4f}")
    line = " | ".join(parts)
    logger.info(line)


def compute_metrics(log_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.compute() if isinstance(value, Metric) else value
        for name, value in log_dict.items()
    }


def reset_metrics(log_dict: dict[str, Any]) -> None:
    for value in log_dict.values():
        if isinstance(value, Metric):
            value.reset()


def get_checkpoint_args(
    checkpoint_args: dict[str, Any] | TaskCheckpointArgs | None,
) -> TaskCheckpointArgs:
    if isinstance(checkpoint_args, TaskCheckpointArgs):
        return checkpoint_args
    checkpoint_args = {} if checkpoint_args is None else checkpoint_args
    args = validate.pydantic_model_validate(TaskCheckpointArgs, checkpoint_args)
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
    logger.info(f"Loading checkpoint from '{ckpt_path}'")
    fabric.load(path=ckpt_path, state=state)  # type: ignore[arg-type]
