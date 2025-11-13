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
from typing import Any, Literal

import torch

from lightly_train._commands import common_helpers
from lightly_train._env import Env
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

DOWNLOADABLE_MODEL_BASE_URL = (
    "https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com"
)

LIGHTLY_TRAIN_PRETRAINED_MODEL = str

DOWNLOADABLE_MODEL_URL_AND_HASH: dict[str, tuple[str, str]] = {
    "dinov2/vits14-noreg-ltdetr-coco": (
        "dinov2_vits14_noreg_ltdetr_coco_251113_fdd81fc4.pt",
        "fdd81fc445f824b45a36d3db4778354703d706bbc1fc9af0aeba9db74e3fee7f",
    ),
    "dinov2/vits14-ltdetr-dsp-coco": (
        "dinov2_ltdetr_2/ltdetr_vits14dinov2_coco_dsp.pt",
        "7e1f91b251ba0b796d88fb68276a24a52341aa6e8fb40abe9f730c2a093a5b40",
    ),
    "dinov3/convnext-tiny-ltdetr-coco": (
        "dinov3_convnext_tiny_ltdetr_coco_251113_3a90352e.pt",
        "3a90352ed61d9e4fa38ec9b7e2a38a8186a54412a4b0044c391e165e7742db22",
    ),
    "dinov3/convnext-small-ltdetr-coco": (
        "dinov3_convnext_small_ltdetr_coco_251113_a94a28a0.pt",
        "a94a28a0c47640d871c433b38901f941aa7a1676ba925743581ea99876c59835",
    ),
    "dinov3/convnext-base-ltdetr-coco": (
        "dinov3_convnext_base_ltdetr_coco_251113_a310a474.pt",
        "a310a474f04c722bed97fcc185908bbedad524de5a50df1292e1bda2ee23c789",
    ),
    "dinov3/convnext-large-ltdetr-coco": (
        "dinov3_convnext_large_ltdetr_coco_251113_9946dca5.pt",
        "9946dca52dc102994a2705984e0cca05ce2f0337ba8417c0db88c934c6c052bd",
    ),
    "dinov3/vits16-eomt-coco": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cocostuff.pt",
        "5078dd29dc46b83861458f45b6ed94634faaf00bebcd9f0d95c1d808602b1f0c",
    ),
    "dinov3/vitb16-eomt-coco": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cocostuff.pt",
        "721a84dc05176a1def4fa15b5ddb8fd4e284c200c36d8af8d60d7a0704820bc5",
    ),
    "dinov3/vitl16-eomt-coco": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cocostuff.pt",
        "b4b31eaaec5f4ddb1c4e125c3eca18f834841c6d6552976b0c2172ff798fb75a",
    ),
    "dinov3/vits16-eomt-cityscapes": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cityscapes.pt",
        "ef7d54eac202bb0a6707fd7115b689a748d032037eccaa3a6891b57b83f18b7e",
    ),
    "dinov3/vitb16-eomt-cityscapes": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cityscapes.pt",
        "e78e6b1f372ac15c860f64445d8265fd5e9d60271509e106a92b7162096c9560",
    ),
    "dinov3/vitl16-eomt-cityscapes": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cityscapes.pt",
        "3f397e6ca0af4555adb1da9efa489b734e35fbeac15b4c18e408c63922b41f6c",
    ),
    "dinov3/vits16-eomt-ade20k": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vits16_autolabel_sun397.pt",
        "f9f002e5adff875e0a97a3b310c26fe5e10c26d69af4e830a4a67aa7dda330aa",
    ),
    "dinov3/vitb16-eomt-ade20k": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_autolabel_sun397.pt",
        "400f7a1b42a7b67babf253d6aade0be334173d70e7351a01159698ac2d2335ca",
    ),
    "dinov3/vitl16-eomt-ade20k": (
        "dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_ade20k.pt",
        "eb31183c70edd4df8923cba54ce2eefa517ae328cf3caf0106d2795e34382f8f",
    ),
    "dinov3/vits16-eomt-inst-coco": (
        "/dinov3_eomt/dinov3_vits16_eomt_inst_coco.pt",
        "b54dafb12d550958cc5c9818b061fba0d8b819423581d02080221d0199e1cc37",
    ),
    "dinov3/vitb16-eomt-inst-coco": (
        "/dinov3_eomt/dinov3_vitb16_eomt_inst_coco.pt",
        "a57b5e7afd5cd64422d74d400f30693f80f96fa63184960250fb0878afd3c7f6",
    ),
    "dinov3/vitl16-eomt-inst-coco": (
        "/dinov3_eomt/dinov3_vitl16_eomt_inst_coco.pt",
        "1aac5ac16dcbc1a12cc6f8d4541bea5e7940937a49f0b1dcea7394956b6e46e5",
    ),
}


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
        raise ValueError(f"Unknown model name or checkpoint path: '{checkpoint}'")
    return local_ckpt_path


def init_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    # Import the model class dynamically
    module_path, class_name = checkpoint["model_class_path"].rsplit(".", 1)
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


def queries_adjust_num_queries_hook(
    module: torch.Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Resize query embeddings from the checkpoint to match the module configuration."""
    queries_weight_key = f"{prefix}queries.weight"
    queries_weight = state_dict.get(queries_weight_key)
    if queries_weight is None:
        return

    query_embed_module = getattr(module, "queries", None)
    num_queries_module = getattr(module, "num_queries", None)
    if query_embed_module is None or num_queries_module is None:
        return

    num_queries_state = queries_weight.shape[0]
    if num_queries_state == num_queries_module:
        return
    elif num_queries_state > num_queries_module:
        logger.info(
            f"Checkpoint provides {num_queries_state} queries but module expects {num_queries_module}. Truncating.",
        )

        queries_weight = queries_weight[:num_queries_module, :]
    else:
        logger.info(
            f"Checkpoint provides {num_queries_state} queries but module expects {num_queries_module}. Repeating entries.",
        )

        repeated_times, remainder = divmod(num_queries_module, num_queries_state)
        queries_weight = queries_weight.repeat(repeated_times, 1)
        if remainder > 0:
            queries_weight = torch.cat(
                [queries_weight, queries_weight[:remainder, :]], dim=0
            )

    state_dict[queries_weight_key] = queries_weight
