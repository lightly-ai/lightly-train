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

DOWNLOADABLE_MODEL_URL_AND_HASH: dict[str, tuple[str, str]] = {
    "dinov2/vits14-ltdetr-coco": (
        "/dinov2_ltdetr/ltdetr_vits14dinov2_coco.ckpt",
        "bfa2adf2b4dc6527947f9f361e47933959a4e9461efa54e02fd39062cd01aac5",
    ),
    "dinov2/vits14-ltdetr-dsp-coco": (
        "/dinov2_ltdetr/ltdetr_vits14dinov2_coco_dsp.ckpt",
        "3115eeee80f1ff9edae3a3956ec3d7204ca7912453f21a4848954e7c1c73db02",
    ),
    "dinov3/convnext-tiny-ltdetr-coco": (
        "/dinov3_ltdetr/ltdetr_convnext-tiny_coco.ckpt",
        "a976d45a8512c80d88b764f179755b4c91e42b97e7cf7061ddf0283900924aff",
    ),
    "dinov3/convnext-small-ltdetr-coco": (
        "/dinov3_ltdetr/ltdetr_convnext-small_coco.ckpt",
        "509d3de9759950dc72cf53f1a435bb6b0d8a7acf4c4883bd1ee74d8bae27310b",
    ),
    "dinov3/convnext-base-ltdetr-coco": (
        "/dinov3_ltdetr/ltdetr_convnext-base_coco.ckpt",
        "542c788dd1b0eec7873243667a8761a766e87c6922ab59a59896b67ad6d802c3",
    ),
    "dinov3/convnext-large-ltdetr-coco": (
        "/dinov3_ltdetr/ltdetr_convnext-large_coco.ckpt",
        "1c862670adeeb11c7ae4a1e4c422bf3c71df2d3193e0c5c5c6f6ffc640244ae1",
    ),
    "dinov3/vits16-eomt-coco": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cocostuff.pt",
        "5078dd29dc46b83861458f45b6ed94634faaf00bebcd9f0d95c1d808602b1f0c",
    ),
    "dinov3/vitb16-eomt-coco": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cocostuff.pt",
        "721a84dc05176a1def4fa15b5ddb8fd4e284c200c36d8af8d60d7a0704820bc5",
    ),
    "dinov3/vitl16-eomt-coco": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cocostuff.pt",
        "b4b31eaaec5f4ddb1c4e125c3eca18f834841c6d6552976b0c2172ff798fb75a",
    ),
    "dinov3/vits16-eomt-cityscapes": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cityscapes.pt",
        "ef7d54eac202bb0a6707fd7115b689a748d032037eccaa3a6891b57b83f18b7e",
    ),
    "dinov3/vitb16-eomt-cityscapes": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cityscapes.pt",
        "e78e6b1f372ac15c860f64445d8265fd5e9d60271509e106a92b7162096c9560",
    ),
    "dinov3/vitl16-eomt-cityscapes": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cityscapes.pt",
        "3f397e6ca0af4555adb1da9efa489b734e35fbeac15b4c18e408c63922b41f6c",
    ),
    "dinov3/vits16-eomt-ade20k": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vits16_autolabel_sun397.pt",
        "f9f002e5adff875e0a97a3b310c26fe5e10c26d69af4e830a4a67aa7dda330aa",
    ),
    "dinov3/vitb16-eomt-ade20k": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_autolabel_sun397.pt",
        "400f7a1b42a7b67babf253d6aade0be334173d70e7351a01159698ac2d2335ca",
    ),
    "dinov3/vitl16-eomt-ade20k": (
        "/dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_ade20k.pt",
        "eb31183c70edd4df8923cba54ce2eefa517ae328cf3caf0106d2795e34382f8f",
    ),
}


def load_model_from_checkpoint(
    checkpoint: PathLike,
    device: Literal["cpu", "cuda", "mps"] | torch.device | None = None,
) -> TaskModel:
    """
    Either load model from an exported model file (in .pt format) or a checkpoint file (in .ckpt format) or download
    it from our model repository.

    First check if `checkpoint` points to a valid file. If not and `checkpoint` is a `str` try to match that name
    to one of the models in our repository and download it. Downloaded models are cached under the location specified by
    the environment variable `LIGHTLY_TRAIN_MODEL_CACHE_DIR`.

    Args:
        checkpoint:
            Either a path to the exported model/checkpoint file or the name of a model in our model repository.
        device:
            Device to load the model on. If None, the model will be loaded onto a GPU
            (`"cuda"` or `"mps"`) if available, and otherwise fall back to CPU.


    Returns:
        The loaded model.
    """
    device = _resolve_device(device)
    if isinstance(checkpoint, Path) or Path(checkpoint).exists():
        checkpoint = common_helpers.get_checkpoint_path(checkpoint=checkpoint)
    else:
        assert isinstance(checkpoint, str)

        logger.info("No checkpoint file found. Trying to download.")

        if checkpoint not in DOWNLOADABLE_MODEL_URL_AND_HASH:
            raise ValueError(f"No downloadable model named {checkpoint}.")
        model_url, model_hash = DOWNLOADABLE_MODEL_URL_AND_HASH[checkpoint]
        model_url = urllib.parse.urljoin(DOWNLOADABLE_MODEL_BASE_URL, model_url)
        download_dir = Env.LIGHTLY_TRAIN_MODEL_CACHE_DIR.value.expanduser().resolve()
        model_name = os.path.basename(urllib.parse.urlparse(model_url).path)
        checkpoint = download_dir / model_name

        needs_download = True
        if not checkpoint.is_file():
            logger.info("No cached checkpoint file found. Downloading...")
        elif checkpoint_hash(checkpoint) != model_hash:
            logger.info(
                "Cached checkpoint file found but hash is different. Downloading..."
            )
        else:
            needs_download = False

        if needs_download:
            torch.hub.download_url_to_file(url=model_url, dst=str(checkpoint))
            logger.info(f"Downloaded checkpoint. Hash: {checkpoint_hash(checkpoint)}")

    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)

    # Import the model class dynamically
    module_path, class_name = ckpt["model_class_path"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Create model instance
    model: TaskModel = model_class(**ckpt["model_init_args"])
    model.load_train_state_dict(state_dict=ckpt["train_model"])
    model.eval()

    model = model.to(device)
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
