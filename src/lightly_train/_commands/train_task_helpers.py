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
from typing import Any, Generator

from lightning_fabric import Fabric
from lightning_fabric import utilities as fabric_utilities
from torch.utils.data import DataLoader, Dataset

from lightly_train._configs import validate
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train._env import Env
from lightly_train._loggers.task_logger_args import TaskLoggerArgs
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation_train import (
    DINOv2SemanticSegmentationTrain,
    DINOv2SemanticSegmentationTrainArgs,
)
from lightly_train._task_models.task_train_model import (
    TaskTrainModel,
    TaskTrainModelArgs,
)
from lightly_train.types import PathLike, TaskDatasetItem

logger = logging.getLogger(__name__)


@contextmanager
def rank_zero_unshared_only(
    fabric: Fabric, path: PathLike
) -> Generator[None, None, None]:
    """The code under this context manager is only executed by rank zero.

    If the filesystem at path is shared, the code is executed only on global rank zero.
    If the filesystem at path is not shared, the code is executed on every local rank zero.
    """
    is_shared = fabric_utilities.is_shared_filesystem(
        strategy=fabric.strategy, path=path
    )
    local = not is_shared
    with fabric.rank_zero_first(local=local):
        if fabric.is_global_zero or (local and fabric.local_rank == 0):
            yield
        else:
            return


def get_out_dir(
    fabric: Fabric, out: PathLike, resume_interrupted: bool, overwrite: bool
) -> Path:
    out_dir = Path(out).resolve()
    with rank_zero_unshared_only(fabric=fabric, path=out_dir):
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
    logger_args: dict[str, Any] | TaskLoggerArgs | None = None,
) -> TaskLoggerArgs:
    if isinstance(logger_args, TaskLoggerArgs):
        return logger_args
    logger_args = {} if logger_args is None else logger_args
    args = validate.pydantic_model_validate(TaskLoggerArgs, logger_args)
    args.resolve_auto(steps=steps)
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


def _identity(x: Any) -> Any:
    return x


def get_dataset(
    dataset_args: MaskSemanticSegmentationDatasetArgs,
) -> MaskSemanticSegmentationDataset:
    # TODO(Guarin, 07/25): MMAP filenames.
    filenames = list(dataset_args.list_image_filenames())
    dataset_cls = dataset_args.get_dataset_cls()
    return dataset_cls(
        dataset_args=dataset_args,
        image_filenames=filenames,
        # TODO(Guarin, 07/25): Add transforms
        transform=_identity,  # type: ignore[arg-type]
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
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value]


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
    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value]


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
    task_args: TaskTrainModelArgs,
) -> TaskTrainModel:
    assert isinstance(task_args, DINOv2SemanticSegmentationTrainArgs)
    return DINOv2SemanticSegmentationTrain(args=task_args)
