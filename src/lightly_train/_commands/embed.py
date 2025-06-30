#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Sequence, Sized

from omegaconf import DictConfig
from pydantic import ConfigDict, Field, field_validator
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.trainer.connectors.accelerator_connector import (  # type: ignore[attr-defined]
    _PRECISION_INPUT,
)
from torch.utils.data import DataLoader, Dataset

from lightly_train import _logging
from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs import omegaconf_utils, validate
from lightly_train._configs.config import PydanticConfig
from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._embedding.embedding_predictor import EmbeddingPredictor
from lightly_train._embedding.embedding_transform import EmbeddingTransform
from lightly_train._embedding.writers import writer_helpers
from lightly_train._embedding.writers.embedding_writer import EmbeddingWriter
from lightly_train._env import Env
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import DatasetItem, PathLike

logger = logging.getLogger(__name__)


def embed(
    *,
    out: PathLike,
    data: PathLike | Sequence[PathLike],
    checkpoint: PathLike,
    format: str | EmbeddingFormat = "torch",
    image_size: int | tuple[int, int] = (224, 224),
    batch_size: int = 128,
    num_workers: int | Literal["auto"] = "auto",
    accelerator: str | Accelerator = "auto",
    overwrite: bool = False,
    precision: _PRECISION_INPUT = "32-true",
) -> None:
    """Embed images from a model checkpoint.

    See the documentation for more information: https://docs.lightly.ai/train/stable/embed.html

    Args:
        out:
            Filepath where the embeddings will be saved. For example "embeddings.csv".
        data:
            Directory containing the images to embed or a sequence of image directories
            and files.
        checkpoint:
            Path to the LightlyTrain checkpoint file used for embedding. The location of
            the checkpoint depends on the train command. If training was run with
            ``out="out/my_experiment"``, then the last LightlyTrain checkpoint is saved
            to ``out/my_experiment/checkpoints/last.ckpt``.
        format:
            Format of the embeddings. Supported formats are ['csv', 'lightly_csv',
            'torch']. 'torch' is the recommended and most efficient format. Torch
            embeddings can be loaded with ``torch.load(out, weigths_only=True)``.
            Choose 'lightly_csv' if you want to use the embeddings as custom
            embeddings with the Lightly Worker.
        image_size:
            Size to which the images are resized before embedding. If a single integer
            is provided, the image is resized to a square with the given side length.
            If a (height, width) tuple is provided, the image is resized to the given
            height and width. Note that not all models support all image sizes.
        batch_size:
            Number of images per batch.
        num_workers:
            Number of workers for the dataloader. 'auto' automatically  sets the number
            of workers based on the available CPU cores.
        accelerator:
            Hardware accelerator. Can be one of ['cpu', 'gpu', 'tpu', 'ipu', 'hpu',
            'mps', 'auto']. 'auto' will automatically select the best accelerator
            available.
        overwrite:
            Overwrite the output file if it already exists.
        precision:
            Embedding precision. Select '32-true' for full 32-bit precision, or
            'bf16-mixed'/'16-mixed' for mixed precision.
    """
    config = EmbedConfig(**locals())
    embed_from_config(config=config)


def embed_from_config(config: EmbedConfig) -> None:
    # Set up logging.
    _warnings.filter_embed_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(common_helpers.pretty_format_args(args=config.model_dump()))

    logger.info(
        f"Embedding images in '{common_helpers.remove_excessive_args({'data': config.data})}'."
    )
    format = _get_format(format=config.format)
    out_path = common_helpers.get_out_path(out=config.out, overwrite=config.overwrite)
    checkpoint_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)
    writer = writer_helpers.get_writer(format=format, filepath=out_path)
    checkpoint_instance = _get_checkpoint(checkpoint=checkpoint_path)
    transform = _get_transform(
        image_size=config.image_size,
        normalize_args=checkpoint_instance.lightly_train.normalize_args,
    )
    num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers, num_devices_per_node=1
    )
    embedding_model = _get_embedding_model(checkpoint=checkpoint_instance)
    embedding_predictor = EmbeddingPredictor(embedding_model=embedding_model)
    # convert to float to avoid issues when loading 16bit/64bit models
    embedding_predictor = embedding_predictor.float()
    accelerator = common_helpers.get_accelerator(accelerator=config.accelerator)
    trainer = _get_trainer(
        accelerator=accelerator, writer=writer, precision=config.precision
    )
    # Create a temporary file to use as a memory map for dataset items. The
    # file has to exist while the dataset is used.
    with common_helpers.verify_out_dir_equal_on_all_local_ranks(
        out=out_path
    ), common_helpers.get_dataset_temp_mmap_path(data=config.data) as mmap_filepath:
        dataset = common_helpers.get_dataset(
            data=config.data,
            transform=transform,
            mmap_filepath=mmap_filepath,
        )
        dataloader = _get_dataloader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=num_workers,
        )
        trainer.predict(
            model=embedding_predictor,
            dataloaders=dataloader,
            return_predictions=False,
        )
    logger.info(f"Embeddings saved to '{out_path}'.")


def embed_from_dictconfig(config: DictConfig) -> None:
    logger.debug(f"Embedding images with config: {config}")
    config_dict = omegaconf_utils.config_to_dict(config=config)
    embed_cfg = validate.pydantic_model_validate(EmbedConfig, config_dict)
    embed_from_config(config=embed_cfg)


class EmbedConfig(PydanticConfig):
    out: PathLike
    data: PathLike | Sequence[PathLike]
    checkpoint: PathLike
    format: str | EmbeddingFormat = "torch"
    image_size: int | tuple[int, int] = Field(default=(224, 224))
    batch_size: int = 128
    num_workers: int | Literal["auto"] = "auto"
    accelerator: str | Accelerator = "auto"
    overwrite: bool = False
    precision: _PRECISION_INPUT = "32-true"

    # Allow arbitrary field types such as Module, Dataset, Accelerator, ...
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # TODO(Guarin, 10/24): This is needed because OmegaConf doesn't support tuples and
    # parses them as lists. We can also not set `strict=False` on the field because
    # this raises a Pydantic error due to the union type. I openened an issue in the
    # Pydantic repo and we should remove the workaround once the issue is resolved:
    # https://github.com/pydantic/pydantic/issues/10571
    @field_validator("image_size", mode="before")
    @classmethod
    def transform_image_size(
        cls, image_size: int | tuple[int, int] | list[int]
    ) -> int | tuple[int, ...]:
        if isinstance(image_size, list):
            return tuple(image_size)
        return image_size


class CLIEmbedConfig(EmbedConfig):
    out: str
    data: str
    checkpoint: str
    accelerator: str = "auto"

    # CLI should not pass arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=False)


def _get_format(format: EmbeddingFormat | str) -> EmbeddingFormat:
    logger.debug(f"Getting embedding format for '{format}'.")
    try:
        return EmbeddingFormat(format)
    except ValueError:
        raise ValueError(
            f"Invalid embedding format: '{format}'. Valid formats are: "
            f"{sorted([f.value for f in EmbeddingFormat])}"
        )


def _get_transform(
    image_size: int | tuple[int, int],
    normalize_args: NormalizeArgs,
) -> EmbeddingTransform:
    logger.debug(f"Getting embedding transform for image size {image_size}.")
    mean, std = normalize_args.mean, normalize_args.std
    logger.debug(f"Using mean {mean} and std {std} for normalization.")
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    return EmbeddingTransform(
        image_size=image_size,
        mean=mean,
        std=std,
    )


def _get_dataloader(
    dataset: Dataset[DatasetItem],
    batch_size: int,
    num_workers: int,
) -> DataLoader[DatasetItem]:
    logger.debug(
        f"Getting dataloader with batch_size {batch_size} and num_workers {num_workers}."
    )
    if isinstance(dataset, Sized):
        dataset_size = len(dataset)
        if batch_size > dataset_size:
            old_batch_size = batch_size
            batch_size = dataset_size
            logger.warning(
                f"Detected dataset size {dataset_size} and batch size "
                f"{old_batch_size}. Reducing batch size to {batch_size}."
            )
    timeout = Env.LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC.value if num_workers > 0 else 0
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        timeout=timeout,
    )


def _get_checkpoint(checkpoint: Path) -> Checkpoint:
    logger.debug(f"Loading checkpoint from '{checkpoint}'.")
    return Checkpoint.from_path(checkpoint=checkpoint)


def _get_embedding_model(checkpoint: Checkpoint) -> EmbeddingModel:
    return checkpoint.lightly_train.models.embedding_model


def _get_trainer(
    accelerator: str | Accelerator, writer: EmbeddingWriter, precision: _PRECISION_INPUT
) -> Trainer:
    logger.debug(f"Getting trainer with accelerator '{accelerator}'.")
    return Trainer(
        accelerator=accelerator,
        devices=1,
        inference_mode=True,
        callbacks=[writer],
        logger=False,
        enable_checkpointing=False,
        precision=precision,
    )
