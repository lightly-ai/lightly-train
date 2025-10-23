#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from pathlib import Path
from typing import Any, Sequence

from lightning_fabric import Fabric
from lightning_fabric import utilities as fabric_utilities
from torch.utils.data import DataLoader, Dataset

from lightly_train._data import file_helpers
from lightly_train._data.image_dataset import ImageDataset
from lightly_train._env import Env
from lightly_train._task_models.task_model import TaskModel
from lightly_train._transforms.predict_semantic_segmentation_transform import (
    PredictSemanticSegmentationTransform,
)
from lightly_train._transforms.predict_transform import (
    PredictTransform,
)
from lightly_train.types import DatasetItem, PathLike

logger = logging.getLogger(__name__)

TASK_MODEL_TRANSFROM_CLASSES: dict[str, type[PredictTransform]] = {
    "DINOv2EoMTSemanticSegmentation": PredictSemanticSegmentationTransform,
    "DINOv2LinearSemanticSegmentation": PredictSemanticSegmentationTransform,
    "DINOv3EoMTSemanticSegmentation": PredictSemanticSegmentationTransform,
}


def get_out_dir(
    fabric: Fabric,
    out: PathLike,
    overwrite: bool,
) -> Path:
    # Use the same output directory on all ranks. This avoids issues where users
    # accidentally create different directories on each rank, for example with:
    # out=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_global_rank_zero = fabric.broadcast(str(out))
    out_dir = Path(out_global_rank_zero)

    def check_and_create_out_dir() -> None:
        if out_dir.exists():
            if not out_dir.is_dir():
                raise ValueError(f"Output '{out_dir}' is not a directory!")

            dir_not_empty = any(out_dir.iterdir())

            if dir_not_empty and not overwrite:
                raise ValueError(
                    f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite the directory"
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


def get_transform_cls(model_cls_name: str) -> type[PredictTransform]:
    for task_model_cls, transform_cls in TASK_MODEL_TRANSFROM_CLASSES.items():
        if task_model_cls == model_cls_name:
            return transform_cls
    raise ValueError(f"Unsupported model class '{model_cls_name}'.")


def get_transform(
    model: TaskModel,
) -> PredictTransform:
    model_cls_name = model.class_path.split(".")[-1]
    transform_cls = get_transform_cls(model_cls_name)

    transform_args = transform_cls.transform_args_cls.from_model(model)

    return transform_cls(transform_args=transform_args)


def get_dataset(
    data: PathLike | Sequence[PathLike] | Dataset[DatasetItem],
    transform: PredictTransform,
    num_channels: int,
) -> Dataset[DatasetItem]:
    # TODO(Yutong, 10/25): implement mmap file handling
    if isinstance(data, Dataset):
        logger.debug("Using provided dataset.")
        return data

    if isinstance(data, (str, Path)):
        data = Path(data).resolve()
        if not data.exists():
            raise ValueError(f"Data directory '{data}' does not exist!")
        elif not data.is_dir():
            raise ValueError(f"Data path '{data}' is not a directory!")
        elif data.is_dir() and not any(data.iterdir()):
            raise ValueError(f"Data directory '{data}' is empty!")

        filenames = file_helpers.list_image_filenames_from_dir(image_dir=data)
        return ImageDataset(
            image_dir=data,
            image_filenames=list(
                filenames
            ),  # TODO(Yutong, 10/25): implement mmap file handling
            transform=transform,
            num_channels=num_channels,
        )

    elif isinstance(data, Sequence):
        filenames = file_helpers.list_image_filenames_from_iterable(imgs_and_dirs=data)
        return ImageDataset(
            image_dir=None,
            image_filenames=list(
                filenames
            ),  # TODO(Yutong, 10/25): implement mmap file handling
            transform=transform,
            num_channels=num_channels,
        )
    else:
        raise ValueError(
            "Data must be a directory, a list of directories or files, or a dataset."
        )


def get_dataloader(
    fabric: Fabric,
    dataset: Dataset[DatasetItem],
    batch_size: int,
    num_workers: int,
    loader_args: dict[str, Any] | None = None,
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
        num_workers=num_workers,
        timeout=timeout,
    )  # TODO: add a collate_fn
    if loader_args is not None:
        logger.debug(f"Using additional dataloader arguments {loader_args}.")
        # Ignore batch_size from loader_args. It is already handled in
        # get_global_batch_size.
        loader_args.pop("batch_size", None)
        loader_args.pop("num_workers", None)
        dataloader_kwargs.update(**loader_args)

    dataloader = DataLoader(**dataloader_kwargs)

    return fabric.setup_dataloaders(dataloader)  # type: ignore[return-value,no-any-return]
