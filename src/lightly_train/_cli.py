#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Callable

from omegaconf import DictConfig, OmegaConf

import lightly_train
from lightly_train import _logging
from lightly_train._commands import embed, export, extract_video_frames, train
from lightly_train._commands.embed import CLIEmbedConfig
from lightly_train._commands.export import CLIExportConfig
from lightly_train._commands.extract_video_frames import CLIExtractVideoFramesConfig
from lightly_train._commands.train import CLITrainConfig
from lightly_train._env import Env
from lightly_train._models import package_helpers
from lightly_train.errors import ConfigError

logger = logging.getLogger(__name__)

__all__ = ["train", "export", "embed", "extract_video_frames", "cli"]

_HELP_COMMANDS = {"help", "--help", "-h"}
_HELP_MSG = """
    Commands:
        lightly-train train                 Train model with self-supervised learning.
        lightly-train export                Export model from checkpoint.
        lightly-train embed                 Embed images using a trained model.
        lightly-train list_models           List supported models for training.
        lightly-train list_methods          List supported methods for training.
        lightly-train extract_video_frames  Extract frames from videos using ffmpeg.
        lightly-train help                  Show help message.

    Run `lightly-train <command> help` for more information on a specific command.

    Optional arguments:
        -v, --verbose  Run the command in verbose mode for detailed output.

    See the documentation for more information: https://docs.lightly.ai/train/stable/
    """

_train_cfg = CLITrainConfig(out="", data="", model="")
_TRAIN_HELP_MSG = f"""
    Train a model with self-supervised learning.

    See the documentation for more information: https://docs.lightly.ai/train/stable/train.html

    The training process can be monitored with TensorBoard:

        tensorboard --logdir out


    After training, the model is exported in the library default format to 
    `out/exported_models/exported_last.pt`. It can be exported to different formats
    using the ``lightly_train.export`` command.

    Usage:
        lightly-train train [options]

    Options:
        out (str, required):
            Output directory to save logs, checkpoints, and other artifacts.
        data (str, required):
            Path to a directory containing images or a sequence of image directories and
            files.
        model (str, required):
            Model name for training. For example 'torchvision/resnet50'.
            Run `lightly-train list_models` to see all supported models.
        method (str, required):
            Method name for training. For example 'simclr'. Default: {_train_cfg.method}
            Run `lightly-train list_methods` to see all supported methods.
        method_args (dict):
            Arguments for the self-supervised learning method. The available arguments
            depend on the `method` parameter.
        embed_dim (int):
            Embedding dimension. Set this if you want to train an embedding model with
            a specific dimension. By default, the output dimension of `model` is used.
        epochs (int):
            Number of training epochs. Default: {_train_cfg.epochs}
        batch_size (int):
            Global batch size. The batch size per device/GPU is inferred from this value
            and the number of devices and nodes. Default: {_train_cfg.batch_size}
        num_workers (int | "auto"):
            Number of workers for the dataloader per device/GPU. 'auto' automatically  
            sets the number of workers based on the available CPU cores. Default: {_train_cfg.num_workers}
        devices (int):
            Number of devices/GPUs for training. 'auto' automatically selects all
            available devices. The device type is determined by the `accelerator`
            parameter. Default: {_train_cfg.devices}
        num_nodes (int):
            Number of nodes for distributed training. Default: {_train_cfg.num_nodes}
        checkpoint (str):
            Use this parameter to further pretrain a model from a previous run.
            The checkpoint must be a path to a checkpoint file created by a previous
            training run, for example "out/my_experiment/checkpoints/last.ckpt".
            This will only load the model weights from the previous run. All other
            training state (e.g. optimizer state, epochs) from the previous run are not
            loaded. Instead, a new run is started with the model weights from the
            checkpoint.

            If you want to resume training from an interrupted or crashed run, use the
            ``resume_interrupted`` parameter instead.
            See https://docs.lightly.ai/train/stable/train/index.html#resume-training
            for more information.
            Default: {_train_cfg.checkpoint}
        resume_interrupted (bool):
            Set this to True if you want to resume training from an **interrupted or
            crashed** training run. This will pick up exactly where the training left
            off, including the optimizer state and the current epoch.

            - You must use the same ``out`` directory as the interrupted run.
            - You must **NOT** change any training parameters (e.g., learning rate, batch size, data, etc.).
            - This is intended for continuing the same run without modification.

            If you want to further pretrain a model or change the training parameters,
            use the ``checkpoint``parameter instead.
            See https://docs.lightly.ai/train/stable/train/index.html#resume-training
            for more information.
            Default: {_train_cfg.resume_interrupted}
        overwrite (bool):
            Overwrite the output directory if it exists. Warning, this might overwrite
            existing files in the directory! Default: {_train_cfg.overwrite}
        accelerator (str):
            Hardware accelerator. Can be one of ['cpu', 'gpu', 'tpu', 'ipu', 'hpu',
            'mps', 'auto']. 'auto' automatically selects the best acccelerator
            available. Default: {_train_cfg.accelerator}
        strategy (str):
            Training strategy. For example 'ddp' or 'auto'. 'auto' automatically
            selects the best strategy available. Default: {_train_cfg.strategy}
        precision (str):
            Training precision. Select '16-mixed' for mixed 16-bit precision, '32-true'
            for full 32-bit precision, or 'bf16-mixed' for mixed bfloat16 precision.
            Default: {_train_cfg.precision}
        float32_matmul_precision (str):
            Precision for float32 matrix multiplication. Can be one of ['auto',
            'highest', 'high', 'medium']. See https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
            for more information.
            Default: {_train_cfg.float32_matmul_precision}
        seed (int):
            Random seed for reproducibility. Default: {_train_cfg.seed}
        loggers (dict):
            Loggers for training. Either null or a dictionary of logger names to either
            null or a dictionary of logger arguments. Null uses the default loggers.
            To disable a logger, set it to null: `loggers.tensorboard=null`.
            To configure a logger, pass the respective arguments:
            `loggers.wandb.project="my_project"`.
            Default: null
        callbacks (dict):
            Callbacks fo training. Either null or a dictionary of callback names to
            either null or a dictionary of callback arguments. Null uses the default
            callbacks. To disable a callback, set it to null:
            `callbacks.model_checkpoint=null`. To configure a callback, pass the
            respective arguments: `callbacks.model_checkpoint.every_n_epochs=5`.
            Default: null
        optim (str):
            Optimizer name. Must be one of ['auto', 'adamw', 'sgd']. 'auto' automatically
            selects the optimizer based on the method.
        optim_args (dict):
            Optimizer arguments. Available arguments depend on the optimizer.
            AdamW:
                - optim_args.lr (float)
                - optim_args.betas (float, float)
                - optim_args.weight_decay (float)
            SGD:
                - optim_args.lr (float)
                - optim_args.momentum (float)
                - optim_args.weight_decay (float)
        transform_args (dict):
            Arguments for the image transform. The available arguments depend on the
            `method` parameter. The following arguments are always available:
            - transform_args.image_size (int, int)
            - transform_args.random_resize.min_scale (float)
            - transform_args.random_resize.max_scale (float)
            - transform_args.random_flip.horizontal_prob (float)
            - transform_args.random_flip.vertical_prob (float)
            - transform_args.random_rotation.prob (float)
            - transform_args.random_rotation.degrees (int)
            - transform_args.random_gray_scale (float)
            - transform_args.normalize.mean (float, float, float)
            - transform_args.normalize.std (float, float, float)
        loader_args (dict):
            Additional arguments for the PyTorch DataLoader.
        trainer_args (dict):
            Additional arguments for the PyTorch Lightning Trainer.
        model_args (dict):
            Arguments for the model. The available arguments depend on the `model`
            parameter. For example, if `model='torchvision/<model_name>'`, the
            arguments are passed to
            `torchvision.models.get_model(model_name, **model_args)`.
        resume (bool):
            Deprecated. Use `resume_interrupted` instead.
            Default: null

    Optional arguments:
        -v, --verbose  Run the command in verbose mode for detailed output.

    Examples:
    # Train a ResNet-18 model with SimCLR on ImageNet
    lightly-train train out=out data=imagenet/train model=torchvision/resnet18 method=simclr

    # Train a ConvNext embedding model with DINO
    lightly-train train out=out data=imagenet/train model=torchvision/convnext_small \\
        method=dino embed_dim=128 epochs=300 batch_size=64 precision=16-mixed \\
        transform_args.global_crop_size=178 optim_args.lr=0.01 \\
        optim_args.betas="[0.9, 0.999]"
"""
_export_cfg = CLIExportConfig(checkpoint="", out="")
_EXPORT_HELP_MSG = f"""
    Export a model from a checkpoint.

    See the documentation for more information: https://docs.lightly.ai/train/stable/export.html

    Usage:
        lightly-train export [options]

    Options:
        out (str, required):
            Path where the exported model will be saved.
        checkpoint (str, required):
            Path to the LightlyTrain checkpoint file to export the model from. The
            location of the checkpoint depends on the train command. If training was run
            with `out="out/my_experiment"`, then the last LightlyTrain checkpoint is
            saved to `out/my_experiment/checkpoints/last.ckpt`.
        part (str):
            Part of the model to export. Valid options are 'model' and
            'embedding_model'. 'model' is the default option and exports the model
            that was passed as `model` argument to the train function.
            'embedding_model' exports the embedding model. This includes the model
            passed with the model argument in the train function and an extra embedding
            layer if the `embed_dim` argument was set during training. This is useful
            if you want to use the exported model for embedding images.
            Default: {_export_cfg.part}
        format (str):
            Format to save the model in. Valid options are ['package_default',
            'torch_model', 'torch_state_dict'].
            'package_default' is the default option and exports the model in the
            default format of the package that was used for training. This ensures
            compatibility with the package and is the most flexible option.
            'torch_state_dict' exports the model's state dict which can be loaded with
            `model.load_state_dict(torch.load(out, weights_only=True))`.
            'torch_model' exports the model as a torch module which can be loaded with
            `model = torch.load(out)`. This requires that the same LightlyTrain version
            is installed when the model is exported and when it is loaded again.
            Default: {_export_cfg.format}
        overwrite (bool):
            Overwrite the output file if it already exists. Default: {_export_cfg.overwrite}

    Optional arguments:
        -v, --verbose  Run the command in verbose mode for detailed output.

    Examples:
    # Export the model in the package default format
    lightly-train export checkpoint=out/checkpoints/last.ckpt out=out/model.pth

    # Export the state dict of the model
    lightly-train export checkpoint=out/checkpoints/last.ckpt out=out/model.pth \\
        format=torch_state_dict

    # Export the embedding model as a torch module
    lightly-train export checkpoint=out/checkpoints/last.ckpt out=out/embedding_model.pth \\
        part=embedding_model format=torch_model
"""
_embed_cfg = CLIEmbedConfig(out="", data="", checkpoint="")
_EMBED_HELP_MSG = f"""
    Embed images from a model checkpoint.

    See the documentation for more information: https://docs.lightly.ai/train/stable/embed.html

    Usage:
        lightly-train embed [options]

    Options:
        out (str, required):
            Filepath where the embeddings will be saved. For example "embeddings.csv".
        data (str, required):
            Directory containing the images to embed or a sequence of image directories
            and files.
        checkpoint (str, required):
            Path to the LightlyTrain checkpoint file used for embedding. The location of
            the checkpoint depends on the train command. If training was run with
            `out="out/my_experiment"`, then the last LightlyTrain checkpoint is saved to
            `out/my_experiment/checkpoints/last.ckpt`.
        format (str, required):
            Format of the embeddings. Supported formats are ['csv', 'lightly_csv',
            'torch']. 'torch' is the recommended and most efficient format. Torch
            embeddings can be loaded with `torch.load(out, weigths_only=True)`.
            Choose 'lightly_csv' if you want to use the embeddings as custom
            embeddings with the Lightly Worker. Default: {_embed_cfg.format}
        image_size (int or [int, int]):
            Size to which the images are resized before embedding. If a single integer
            is provided, the image is resized to a square with the given side length.
            If a [height, width] list is provided, the image is resized to the given
            height and width. Note that not all models support all image sizes.
            Default: {_embed_cfg.image_size}
        batch_size (int):
            Number of images per batch. Default: {_embed_cfg.batch_size}
        num_workers (int | "auto"):
            Number of workers for the dataloader. 'auto' automatically  sets the number
            of workers based on the available CPU cores. Default: {_embed_cfg.num_workers}
        accelerator (str):
            Hardware accelerator. Can be one of ['cpu', 'gpu', 'tpu', 'ipu', 'hpu',
            'mps', 'auto']. 'auto' will automatically select the best accelerator
            available. Default: {_embed_cfg.accelerator}
        overwrite (bool):
            Overwrite the output file if it already exists. Default: {_embed_cfg.overwrite}
        precision (str):
            Embedding precision. Select '32-true' for full 32-bit precision, or 
            'bf16-mixed'/'16-mixed' for mixed precision. Default: {_embed_cfg.precision}

    Optional arguments:
        -v, --verbose  Run the command in verbose mode for detailed output.

    Examples:
    # Embed images from a model checkpoint
    lightly-train embed out=embeddings.csv data=images checkpoint=out/checkpoints/last.ckpt \\
        format=csv

    # Create custom embeddings for the Lightly Worker
    lightly-train embed out=embeddings.csv data=images checkpoint=out/checkpoints/last.ckpt \\
        format=lightly_csv

    # Embed images with a different image size
    lightly-train embed out=embeddings.csv data=images checkpoint=out/checkpoints/last.ckpt \\
        format=csv image_size="[448, 672]"
"""

_extract_cfg = CLIExtractVideoFramesConfig(data="", out="")
_EXTRACT_VIDEO_FRAMES_HELP_MSG = f"""
    Extract frames from videos using ffmpeg.

    Directly calls ffmpeg via subprocess. This is the most performant option. Requires
    ffmpeg to be installed on the system.
    Installation of ffmpeg:
        - {extract_video_frames.FFMPEG_INSTALLATION_EXAMPLES[0]}
        - {extract_video_frames.FFMPEG_INSTALLATION_EXAMPLES[1]}
        - {extract_video_frames.FFMPEG_INSTALLATION_EXAMPLES[2]}

    Usage:
        lightly-train extract_video_frames [options]

    Options:
        data (str, required):
            Path to a directory containing video files.
        out (str, required):
            Output directory to save the extracted frames.
        overwrite (bool):
            If True, existing frames are overwritten. If false, the out directory must
            be empty. Default: {_extract_cfg.overwrite}
        frame_filename_format (str):
            Filename format for the extracted frames, passed as it is to ffmpeg.
            Default: "{_extract_cfg.frame_filename_format}" for extracting frames as jpg
            files and with the 9-digit frame number as filename.
        num_workers (int | "auto"):
            Number of parallel calls to ffmpeg. 'auto' automatically sets the number of
            workers based on the available CPU cores. Default: {_extract_cfg.num_workers}

    Optional arguments:
        -v, --verbose  Run the command in verbose mode for detailed output.

    Examples:
    # Extract frames from videos
    lightly-train extract_video_frames data=videos out=frames

    # Extract frames with a custom filename format
    lightly-train extract_video_frames data=videos out=frames frame_filename_format="%04d.jpg"

    # Extract frames using 2 parallel calls to ffmpeg
    lightly-train extract_video_frames data=videos out=frames frame_filename_format="%04d.jpg"
"""


_VERBOSE_FLAGS = ["-v", "--verbose"]


def cli(config: DictConfig) -> None:
    keys = list(config.keys())

    # Check if the user wants to run the command in verbose mode.
    # Any of the following will enable verbose mode: -v, --verbose
    if any(flag in keys for flag in _VERBOSE_FLAGS):
        os.environ[Env.LIGHTLY_TRAIN_LOG_LEVEL.name] = logging.getLevelName(
            logging.DEBUG
        )
        config = OmegaConf.create(
            {k: v for k, v in config.items() if k not in _VERBOSE_FLAGS}
        )
    _logging.set_up_console_logging()

    if config.is_empty():
        _show_help()
        return

    # First argument after lightly_train is the command. For example `lightly-train train ...`
    command = str(keys[0]).lower()
    help_if_config_empty = True
    if command in _HELP_COMMANDS:
        _show_help()
        return
    elif command == "train":
        command_fn = train.train_from_dictconfig
        help_msg = _TRAIN_HELP_MSG
    elif command == "export":
        command_fn = export.export_from_dictconfig
        help_msg = _EXPORT_HELP_MSG
    elif command == "embed":
        command_fn = embed.embed_from_dictconfig
        help_msg = _EMBED_HELP_MSG
    elif command == "extract_video_frames":
        command_fn = extract_video_frames.extract_video_frames_from_dictconfig
        help_msg = _EXTRACT_VIDEO_FRAMES_HELP_MSG
    elif command == "list_models":
        command_fn = _list_models
        help_msg = ""
        help_if_config_empty = False
    elif command == "list_methods":
        command_fn = _list_methods
        help_msg = ""
        help_if_config_empty = False
    else:
        _show_invalid_command_help(command=command)
        sys.exit(1)

    config.pop(command)
    _run_command_fn(
        command_fn=command_fn,
        config=config,
        help_msg=help_msg,
        help_if_config_empty=help_if_config_empty,
    )


def _cli_entrypoint() -> None:
    # Entrypoint to CLI used in pyproject.toml
    cli(config=OmegaConf.from_cli())


def _run_command_fn(
    command_fn: Callable[[DictConfig], None],
    config: DictConfig,
    help_msg: str,
    help_if_config_empty: bool,
) -> None:
    """Runs a subcommand function with the given config.

    Args:
        command_fn:
            The function to run.
        config:
            Config passed to `command_fn`.
        help_msg:
            The help message to display if a help command is found in the config. For
            example in `lightly-train train help`.
        help_if_config_empty:
            If yes, then show the help message if the config is empty. This is useful
            if a user runs `lightly-train train` without any arguments.
    """
    if _is_help_command_in_config(config) or (
        config.is_empty() and help_if_config_empty
    ):
        _show_msg(help_msg)
        return

    try:
        command_fn(config)
    except ConfigError as ex:
        logger.error(ex)
        raise ex from None  # Shorten stacktrace
    except Exception as ex:
        logger.error(ex)
        raise ex from None  # Shorten stacktrace


def _list_models(config: DictConfig) -> None:
    lines = [f"    {model}" for model in package_helpers.list_model_names()]
    logger.info("\n".join(lines))


def _list_methods(config: DictConfig) -> None:
    lines = [f"    {method}" for method in lightly_train.list_methods()]
    logger.info("\n".join(lines))


def _is_help_command_in_config(config: DictConfig) -> bool:
    return any(help_command in config for help_command in _HELP_COMMANDS)


def _show_help() -> None:
    _show_msg(_HELP_MSG)


def _show_invalid_command_help(command: str) -> None:
    msg = _format_msg(
        f"""
        Unknown command '{command}':
            lightly-train {command}
        """
    )
    msg += "\n"
    msg += _format_msg(_HELP_MSG.replace("Commands:", "Valid commands are:"))
    _show_msg(msg)


def _show_msg(msg: str) -> None:
    logger.info(_format_msg(msg))


def _format_msg(msg: str) -> str:
    # Inspect.cleandoc removes leading whitespaces from messages. This helps with
    # multiline strings.
    return inspect.cleandoc(msg)
