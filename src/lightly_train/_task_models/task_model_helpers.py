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
from torch.nn import Module, ModuleList

from lightly_train._commands import common_helpers
from lightly_train._env import Env
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

DOWNLOADABLE_MODEL_BASE_URL = (
    "https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com"
)

LIGHTLY_TRAIN_PRETRAINED_MODEL = str

# How to add a new downloadable model:
# 1. Get hash of exported model file with `sha256sum best.pt`
# 2. Upload the exported model file to the S3 bucket and follow the naming scheme:
#    "<package>_<model_name>_[<task>]_<dataset>_[<resolution>]_<date>_<hash>.pt"
#    Example: dinov3_vitt16_ltdetr_coco_251205_1a4c20a1.pt
# 3. Add an entry to the DOWNLOADABLE_MODEL_URL_AND_HASH dictionary below including the
#    model name, file name, and hash.
DOWNLOADABLE_MODEL_URL_AND_HASH: dict[str, tuple[str, str]] = {
    "dinov2/vits14-noreg-ltdetr-coco": (
        "dinov2_vits14_noreg_ltdetr_coco_251218_4e1f523d.pt",
        "4e1f523db68c94516ee5b35a91f24267657af474bea58b52a7f7e51ec2d8f717",
    ),
    "dinov2/vits14-ltdetr-dsp-coco": (
        "dinov2_vits14_ltdetr_dsp_coco_251218_fa435184.pt",
        "fa435184c775205469056f46456941ea271266ee522c656642853d061317f8ae",
    ),
    "dinov3/vitt16-ltdetr-coco": (
        "dinov3_vitt16_ltdetr_coco_251218_dfd34210.pt",
        "dfd34210a1a3375793d149a55d9b49e6e8b783458bdd4cd76fd28fa2d61dbb37",
    ),
    "dinov3/vitt16plus-ltdetr-coco": (
        "dinov3_vitt16plus_ltdetr_coco_251218_af499c82.pt",
        "af499c825436013098a77a028ff5cf08dbf31118f4d68b15eefa6fdd9635f5d2",
    ),
    "dinov3/vits16-ltdetr-coco": (
        "dinov3_vits16_ltdetr_coco_251218_4812416b.pt",
        "4812416b861a80f305889cf1408775044c8b05f1baf9be45cd4b1d0edd5d4532",
    ),
    "dinov3/convnext-tiny-ltdetr-coco": (
        "dinov3_convnext_tiny_ltdetr_coco_251218_35bbc4fb.pt",
        "35bbc4fbec3bb9fa113a33f1013abaab1952edf3335f98624b5914812d63d26c",
    ),
    "dinov3/convnext-small-ltdetr-coco": (
        "dinov3_convnext_small_ltdetr_coco_251218_8f7109ab.pt",
        "8f7109ab406aa92791e4e4ca6249ab9a863734795676c81b91dbd4cc4b1ef387",
    ),
    "dinov3/convnext-base-ltdetr-coco": (
        "dinov3_convnext_base_ltdetr_coco_251218_836adb6b.pt",
        "836adb6b5122665a24b6da3ee1720b9f3d0fc3c30cee44cfbd98dcb79fe0809a",
    ),
    "dinov3/convnext-large-ltdetr-coco": (
        "dinov3_convnext_large_ltdetr_coco_251218_03fe6750.pt",
        "03fe6750392daf3ecd32bbab3f144bd5c4d6cdc8bd75635f9e1c5e296e7dd8b0",
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
    # Trained with 2x schedule (180k steps)
    "dinov3/vits16-eomt-panoptic-coco": (
        "dinov3_vits16_eomt_panoptic_coco_251219_89e8a64f.pt",
        "89e8a64fb601c509df76d09ed6ddb6789e080147cadcff9700cf5792dfc20167",
    ),
    # Trained with 2x schedule (180k steps)
    "dinov3/vitb16-eomt-panoptic-coco": (
        "dinov3_vitb16_eomt_panoptic_coco_251209_05948298.pt",
        "0594829822a23935079c35304f3bd1c7fede802114bc1a699780df693f2dea6c",
    ),
    "dinov3/vitl16-eomt-panoptic-coco": (
        "dinov3_vitl16_eomt_panoptic_coco_251209_e0c1e6ae.pt",
        "e0c1e6aeb245dbe6fd8735ffea48b81978b66b1a320533498de4375c18ad4368",
    ),
    "dinov3/vitl16-eomt-panoptic-coco-1280": (
        "dinov3_vitl16_eomt_panoptic_coco_1280_251209_3da0b210.pt",
        "3da0b21000bba3747bcb3e4ac4ee1e38641614022281f4b710d7442c643182f2",
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
    module: Module,
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


def denoising_class_embed_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    denoising_class_embed_weight_key = f"{prefix}denoising_class_embed.weight"
    denoising_class_embed_weight = state_dict.get(denoising_class_embed_weight_key)
    if denoising_class_embed_weight is None:
        return

    denoising_class_embed_module = getattr(module, "denoising_class_embed", None)
    if denoising_class_embed_module is None:
        return

    num_classes_state = denoising_class_embed_weight.shape[0]
    num_classes_module = denoising_class_embed_module.num_embeddings
    if num_classes_state == num_classes_module:
        return
    else:
        logger.info(
            f"Checkpoint provides {num_classes_state - 1} classes but module expects {num_classes_module - 1}. Reinitializing denoising class embed.",
        )
        # Keep the module initialization by overwriting the checkpoint weights with the
        # current parameter tensors.
        state_dict[denoising_class_embed_weight_key] = (
            denoising_class_embed_module.weight.detach().clone()
        )


def class_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    class_head_weight_key = f"{prefix}class_head.weight"
    class_head_bias_key = f"{prefix}class_head.bias"
    class_head_weight = state_dict.get(class_head_weight_key)
    if class_head_weight is None:
        return

    class_head_module = getattr(module, "class_head", None)
    if class_head_module is None:
        return

    num_classes_state = class_head_weight.shape[0]
    num_classes_module = class_head_module.out_features
    if num_classes_state == num_classes_module:
        return
    else:
        logger.info(
            f"Checkpoint provides {num_classes_state - 1} classes but module expects {num_classes_module - 1}. Reinitializing class head.",
        )

        # Keep the module initialization by overwriting the checkpoint weights with the
        # current parameter tensors.
        state_dict[class_head_weight_key] = class_head_module.weight.detach().clone()
        state_dict[class_head_bias_key] = class_head_module.bias.detach().clone()


def score_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    _score_head_reuse_or_reinit_hook(
        module,
        state_dict,
        prefix,
        enc_or_dec="enc",
    )
    _score_head_reuse_or_reinit_hook(
        module,
        state_dict,
        prefix,
        enc_or_dec="dec",
    )


def _score_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    enc_or_dec: Literal["enc", "dec"],
) -> None:
    module_name = f"{enc_or_dec}_score_head"
    score_head_module = getattr(module, module_name, None)
    if score_head_module is None:
        return

    if isinstance(score_head_module, ModuleList):
        for idx, head_module in enumerate(score_head_module):
            is_reinit = _reuse_or_reinit(
                head_module,
                state_dict,
                weight_key=f"{prefix}{module_name}.{idx}.weight",
                bias_key=f"{prefix}{module_name}.{idx}.bias",
            )
    else:
        is_reinit = _reuse_or_reinit(
            score_head_module,
            state_dict,
            weight_key=f"{prefix}{module_name}.weight",
            bias_key=f"{prefix}{module_name}.bias",
        )

    if is_reinit:
        logger.info(
            f"Checkpoint provides different number of classes for {module_name}. Reinitializing score head.",
        )


def _reuse_or_reinit(
    head_module: Module,
    state_dict: dict[str, Any],
    *,
    weight_key: str,
    bias_key: str,
) -> bool:
    score_head_weight = state_dict.get(weight_key)
    if score_head_weight is None:
        return False

    num_classes_state = score_head_weight.shape[0]
    out_features = getattr(head_module, "out_features", None)
    if out_features is None or num_classes_state == out_features:
        return False

    # Keep the module initialization by overwriting the checkpoint weights with the
    # current parameter tensors.
    state_dict[weight_key] = head_module.weight.detach().clone()  # type: ignore[operator]
    state_dict[bias_key] = head_module.bias.detach().clone()  # type: ignore[operator]

    return True
