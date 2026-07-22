#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import hashlib
import importlib
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Literal, NoReturn

import torch

from lightly_train._commands import common_helpers
from lightly_train._configs.model_registry import ModelRegistry
from lightly_train._env import Env
from lightly_train._task_models.dinov3_eomt_instance_segmentation.config import (
    DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY,
)
from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.config import (
    DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.config import (
    DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY,
)
from lightly_train._task_models.ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
)
from lightly_train._task_models.picodet_object_detection.config import (
    PICODET_OBJECT_DETECTION_MODEL_REGISTRY,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

DOWNLOADABLE_MODEL_BASE_URL = (
    "https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com"
)

LIGHTLY_TRAIN_PRETRAINED_MODEL = str

_LEGACY_DINOV2_LTDETR_OBJECT_DETECTION_CLASS_PATH = (
    "lightly_train._task_models.dinov2_ltdetr_object_detection.task_model"
    ".DINOv2LTDETRObjectDetection"
)
_LEGACY_DINOV2_LTDETR_DSP_OBJECT_DETECTION_CLASS_PATH = (
    "lightly_train._task_models.dinov2_ltdetr_object_detection.task_model"
    ".DINOv2LTDETRDSPObjectDetection"
)
_LEGACY_DINOV3_LTDETR_OBJECT_DETECTION_CLASS_PATH = (
    "lightly_train._task_models.dinov3_ltdetr_object_detection.task_model"
    ".DINOv3LTDETRObjectDetection"
)
_GENERIC_LTDETR_OBJECT_DETECTION_CLASS_PATH = (
    "lightly_train._task_models.ltdetr_object_detection.task_model"
    ".LTDETRObjectDetection"
)


def _get_downloadable_model_url_and_hashes(
    registry: ModelRegistry[Any],
) -> dict[str, tuple[str, str]]:
    downloadable_model_url_and_hashes = {}
    for alias in registry.list_aliases():
        try:
            checkpoint = registry.get_alias_metadata(
                alias=alias
            ).downloadable_checkpoint
        except KeyError:
            continue
        downloadable_model_url_and_hashes[alias] = (checkpoint.url, checkpoint.sha256)
    return downloadable_model_url_and_hashes


# How to add a new downloadable model:
# 1. Get hash of exported model file with `sha256sum best.pt`
# 2. Upload the exported model file to the S3 bucket and follow the naming scheme:
#    "<package>_<model_name>_[<task>]_<dataset>_[<resolution>]_<date>_<hash>.pt"
#    Example: dinov3_vitt16_ltdetr_coco_251205_1a4c20a1.pt
# 3. Add an entry to the DOWNLOADABLE_MODEL_URL_AND_HASH dictionary below including the
#    model name, file name, and hash.
DOWNLOADABLE_MODEL_URL_AND_HASH: dict[str, tuple[str, str]] = {
    #### Depth Estimation
    "dinov2/dav3-relative-small": (
        "dinov2_dav3_relative_small_260710_dcc2463f.pt",
        "dcc2463f7fa07606cb1352236889e636a10cc3db64ec31a227a20cc88ce6c21d",
    ),
    "dinov2/dav3-relative-large": (
        "dinov2_dav3_relative_large_260629_9c2e9320.pt",
        "9c2e932085843bbd960e16bc80917b6591e99fc6fd3907ded7bda68d35368e49",
    ),
    "dinov2/dav3-metric-large": (
        "dinov2_dav3_metric_large_260629_6fd208f2.pt",
        "6fd208f22eaccf9007e9e67fb9cad95cc47016c8d00bc74c7fe69ec34185c06b",
    ),
    "dinov2/dav3-metric-small": (
        "dinov2_dav3_metric_small_260713_96a7cd93.pt",
        "96a7cd93ea7175b49bf83f061c76e1e61a807358552b79b5da62f4139b9e862a",
    ),
    "dinov3/dav3-relative-tiny-plus": (
        "dinov3_dav3_relative_tiny_plus_260713_5bff49b8.pt",
        "5bff49b8b07810cd0b6f1551a5be85538a2eab1d0aaf9f2a34ab3bb2124a48d0",
    ),
    "dinov3/dav3-metric-tiny-plus": (
        "dinov3_dav3_metric_tiny_plus_260714_c7b1e414.pt",
        "c7b1e4143d63c73eb0bbdf40e3d94d77f1cc4af027fe223fdeb6f97256d7f964",
    ),
    "dinov3/dav3-metric-tiny": (
        "dinov3_dav3_metric_tiny_260716_111dd31c.pt",
        "111dd31cd8d19caaaaeca92ba109e5f01f6ff02293386e0c42e30d035ec590a2",
    ),
    "dinov3/dav3-relative-tiny": (
        "dinov3_dav3_relative_tiny_260714_90a26f4b.pt",
        "90a26f4bfadc24d30192094c3f4dc52852c70a7f15ceec95b9d303cec3ea1647",
    ),
    # Only the Apache-2.0 Depth Anything V2 models are hosted. The CC-BY-NC-4.0 models
    # (relative base/large and the non-small metric variants) are not redistributed:
    # convert them locally with convert_checkpoint_dav2 and pass the result via
    # `weights=`.
    "dinov2/dav2-relative-small": (
        "dinov2_dav2_relative_small_260629_bb09402a.pt",
        "bb09402aca18dab407707254967b7a1b3cec3dc3707777697ce6101db15d6172",
    ),
    "dinov2/dav2-metric-small-hypersim": (
        "dinov2_dav2_metric_small_hypersim_260629_d5957701.pt",
        "d59577016e01635c285fac76f44685d7a0878545e0b8d560da45c0cf4d058548",
    ),
}
DOWNLOADABLE_MODEL_URL_AND_HASH.update(
    _get_downloadable_model_url_and_hashes(PICODET_OBJECT_DETECTION_MODEL_REGISTRY)
)
DOWNLOADABLE_MODEL_URL_AND_HASH.update(
    _get_downloadable_model_url_and_hashes(
        DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY
    )
)
DOWNLOADABLE_MODEL_URL_AND_HASH.update(
    _get_downloadable_model_url_and_hashes(
        DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY
    )
)
DOWNLOADABLE_MODEL_URL_AND_HASH.update(
    _get_downloadable_model_url_and_hashes(
        DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY
    )
)
DOWNLOADABLE_MODEL_URL_AND_HASH.update(
    _get_downloadable_model_url_and_hashes(LTDETR_MODEL_REGISTRY)
)


def load_model(
    model: PathLike,
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    """Either load model from an exported model file (in .pt format) or a checkpoint file
    (in .ckpt format) or download it from the Lightly model repository.

    First check if `model` points to a valid file. If not and `model` is a `str` try to
    match that name to one of the models in the Lightly model repository and download it.
    Downloaded models are cached under the location specified by the environment variable
    `LIGHTLY_TRAIN_MODEL_CACHE_DIR`.

    Args:
        model:
            Either a path to the exported model/checkpoint file or the name of a model
            in the Lightly model repository.
        device:
            Device to load the model on. If None, the model will be loaded onto a GPU
            (`"cuda"` or `"mps"`) if available, and otherwise fall back to CPU.

    Returns:
        The loaded model.
    """
    device = _resolve_device(device)
    ckpt_path = download_checkpoint(checkpoint=model)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model_instance = init_model_from_checkpoint(checkpoint=ckpt, device=device)
    return model_instance


def load_model_from_checkpoint(
    checkpoint: PathLike,
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    """Deprecated. Use `load_model` instead."""
    return load_model(model=checkpoint, device=device)


def download_checkpoint(checkpoint: PathLike) -> Path:
    """Downloads a checkpoint and returns the local path to it.

    Supports checkpoints from:
    - Local file path
    - Predefined downloadable model names from our repository

    Returns:
        Path to the local checkpoint file.
    """
    ckpt_str = str(checkpoint)
    ckpt_path = Path(checkpoint).resolve()
    if ckpt_path.exists():
        # Local path
        local_ckpt_path = common_helpers.get_checkpoint_path(checkpoint=ckpt_path)
    elif ckpt_str in DOWNLOADABLE_MODEL_URL_AND_HASH:
        # Checkpoint name
        model_url, model_hash = DOWNLOADABLE_MODEL_URL_AND_HASH[ckpt_str]
        model_url = urllib.parse.urljoin(DOWNLOADABLE_MODEL_BASE_URL, model_url)
        download_dir = Env.LIGHTLY_TRAIN_MODEL_CACHE_DIR.value.expanduser().resolve()
        model_name = os.path.basename(urllib.parse.urlparse(model_url).path)
        local_ckpt_path = download_dir / model_name

        needs_download = True
        if not local_ckpt_path.is_file():
            logger.info(
                f"No cached checkpoint file found. Downloading from '{model_url}'..."
            )
        elif checkpoint_hash(local_ckpt_path) != model_hash:
            logger.info(
                "Cached checkpoint file found but hash is different. Downloading from "
                f"'{model_url}'..."
            )
        else:
            needs_download = False

        if needs_download:
            download_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url=model_url, dst=str(local_ckpt_path))
            logger.info(
                f"Downloaded checkpoint to '{local_ckpt_path}'. Hash: "
                f"{checkpoint_hash(local_ckpt_path)}"
            )
    else:
        _raise_unknown_checkpoint_error(checkpoint=checkpoint)
    return local_ckpt_path


def init_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    # Import the model class dynamically. Legacy DINOv2 LT-DETR checkpoints are
    # loaded through the generic LT-DETR implementation after the package deletion.
    model_class_path = checkpoint["model_class_path"]
    if model_class_path == _LEGACY_DINOV2_LTDETR_OBJECT_DETECTION_CLASS_PATH:
        model_class_path = _GENERIC_LTDETR_OBJECT_DETECTION_CLASS_PATH
    elif model_class_path == _LEGACY_DINOV2_LTDETR_DSP_OBJECT_DETECTION_CLASS_PATH:
        raise ValueError(
            "DINOv2 LT-DETR DSP checkpoints are not supported by the generic "
            "LT-DETR object detection model."
        )
    elif model_class_path == _LEGACY_DINOV3_LTDETR_OBJECT_DETECTION_CLASS_PATH:
        model_class_path = _GENERIC_LTDETR_OBJECT_DETECTION_CLASS_PATH
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    model_init_args = checkpoint["model_init_args"]
    model_init_args["load_weights"] = False

    # Create model instance
    model: TaskModel = model_class(**model_init_args)
    model = model.to(device)
    model.load_train_state_dict(state_dict=checkpoint["train_model"])
    model.eval()
    return model


def checkpoint_hash(path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(4096):
            sha256_hash.update(block)
    return sha256_hash.hexdigest().lower()


def _raise_unknown_checkpoint_error(checkpoint: PathLike) -> NoReturn:
    """Raises a helpful error for an unknown model name or checkpoint path.

    Non-hosted Depth Anything V2 models are recognized but not redistributed; for those
    a verbose message points to the local converter. Everything else raises the generic
    error.
    """
    # Imported lazily: this module imports `task_model_helpers` at module level, so a
    # top-level import here would be circular. This runs only on the error path.
    from lightly_train._task_models.depth_estimation.task_model import (
        DepthAnythingDepthEstimation,
    )

    ckpt_str = str(checkpoint)
    # The depth task model's registry is the source of truth for known names. Hosted
    # DAv2 models are matched earlier against DOWNLOADABLE_MODEL_URL_AND_HASH, so a known
    # DAv2 name reaching here is a non-commercial variant that must be converted locally.
    # Filter to DAv2 names: the registry now also holds DAv3 names, which must not trigger
    # the DAv2 non-commercial message.
    dav2_model_names = {
        name.lower()
        for name in DepthAnythingDepthEstimation.list_model_names()
        if "dav2" in name.lower()
    }
    if ckpt_str.lower() in dav2_model_names:
        raise ValueError(
            f"'{ckpt_str}' is a Depth Anything V2 model with a non-commercial license, "
            "which LightlyTrain does not host. Make sure you understand its license "
            "and how it applies to your use, then convert it locally and load the "
            "checkpoint by its path:\n"
            "  python -m lightly_train._task_models.depth_estimation_components"
            f".convert_checkpoint_dav2 --model-name {ckpt_str} --out <converted.pt>\n"
            "The converter reports the license and which official weights to download. "
            'Then call lightly_train.load_model("<converted.pt>").'
        )
    raise ValueError(f"Unknown model name or checkpoint path: '{checkpoint}'")


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the device to load the model on."""
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    elif device is None:
        if torch.cuda.is_available():
            # Return the default CUDA device if available.
            return torch.device("cuda")
        elif device is None and torch.backends.mps.is_available():
            # Return the default MPS device if available.
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        raise ValueError(
            f"Invalid device: {device}. Must be 'cpu', 'cuda', 'mps', a torch.device, or None."
        )
