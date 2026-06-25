#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Convert official Depth Anything V2 weights into a LightlyTrain checkpoint.
Example:
python -m lightly_train._task_models.depth_estimation_components.convert_checkpoint_dav2 --model-name dinov2/dav2-relative-small --out ckpt/dav2_relative_small.pt
python -m lightly_train._task_models.depth_estimation_components.convert_checkpoint_dav2 --model-name dinov2/dav2-metric-large-hypersim --out ckpt/dav2_metric_hypersim_large.pt
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.depth_estimation.task_model import (
    DepthAnythingDepthEstimation,
)

logger = logging.getLogger(__name__)

# Official Hugging Face weights per (parsed) model name.
_HF_WEIGHTS: dict[str, dict[str, str]] = {
    "dinov2/dav2-relative-small": {
        "repo_id": "depth-anything/Depth-Anything-V2-Small",
        "filename": "depth_anything_v2_vits.pth",
    },
    "dinov2/dav2-relative-base": {
        "repo_id": "depth-anything/Depth-Anything-V2-Base",
        "filename": "depth_anything_v2_vitb.pth",
    },
    "dinov2/dav2-relative-large": {
        "repo_id": "depth-anything/Depth-Anything-V2-Large",
        "filename": "depth_anything_v2_vitl.pth",
    },
    "dinov2/dav2-metric-small-hypersim": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        "filename": "depth_anything_v2_metric_hypersim_vits.pth",
    },
    "dinov2/dav2-metric-base-hypersim": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Base",
        "filename": "depth_anything_v2_metric_hypersim_vitb.pth",
    },
    "dinov2/dav2-metric-large-hypersim": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-Hypersim-Large",
        "filename": "depth_anything_v2_metric_hypersim_vitl.pth",
    },
    "dinov2/dav2-metric-small-vkitti": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-VKITTI-Small",
        "filename": "depth_anything_v2_metric_vkitti_vits.pth",
    },
    "dinov2/dav2-metric-base-vkitti": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-VKITTI-Base",
        "filename": "depth_anything_v2_metric_vkitti_vitb.pth",
    },
    "dinov2/dav2-metric-large-vkitti": {
        "repo_id": "depth-anything/Depth-Anything-V2-Metric-VKITTI-Large",
        "filename": "depth_anything_v2_metric_vkitti_vitl.pth",
    },
}

# Task model class per (parsed) model name. The relative and metric models share the
# official checkpoint layout and DPT shapes, so the same key remapping works for both.
_MODEL_CLASSES: dict[str, type[DepthAnythingDepthEstimation]] = {
    "dinov2/dav2-relative-small": DepthAnythingDepthEstimation,
    "dinov2/dav2-relative-base": DepthAnythingDepthEstimation,
    "dinov2/dav2-relative-large": DepthAnythingDepthEstimation,
    "dinov2/dav2-metric-small-hypersim": DepthAnythingDepthEstimation,
    "dinov2/dav2-metric-base-hypersim": DepthAnythingDepthEstimation,
    "dinov2/dav2-metric-large-hypersim": DepthAnythingDepthEstimation,
    "dinov2/dav2-metric-small-vkitti": DepthAnythingDepthEstimation,
    "dinov2/dav2-metric-base-vkitti": DepthAnythingDepthEstimation,
    "dinov2/dav2-metric-large-vkitti": DepthAnythingDepthEstimation,
}

# ``backbone.mask_token`` only exists for masked-image-modeling pretraining and is never
# read during depth inference. The official checkpoint stores it with shape (1, 1, C)
# while this fork's backbone expects (1, C); rather than reshape an unused buffer, the
# remap drops it and we tolerate it being missing, keeping the model's random init.
_ALLOWED_MISSING_OFFICIAL_KEYS = {"backbone.mask_token"}

# License per (parsed) model name. Only the Apache-2.0 models are hosted by LightlyTrain
# and auto-downloaded from Hugging Face. The non-commercial models must be downloaded by
# the user, who is responsible for complying with the model's license terms.
_MODEL_LICENSES: dict[str, str] = {
    "dinov2/dav2-relative-small": "Apache-2.0",
    "dinov2/dav2-relative-base": "CC-BY-NC-4.0",
    "dinov2/dav2-relative-large": "CC-BY-NC-4.0",
    "dinov2/dav2-metric-small-hypersim": "Apache-2.0",
    "dinov2/dav2-metric-base-hypersim": "CC-BY-NC-4.0",
    "dinov2/dav2-metric-large-hypersim": "CC-BY-NC-4.0",
    "dinov2/dav2-metric-small-vkitti": "CC-BY-NC-SA-3.0",
    "dinov2/dav2-metric-base-vkitti": "CC-BY-NC-SA-3.0",
    "dinov2/dav2-metric-large-vkitti": "CC-BY-NC-SA-3.0",
}


def convert_checkpoint(
    out: Path,
    model_name: str = "dinov2/dav2-relative-large",
    weights: Path | None = None,
) -> Path:
    """Convert official Depth Anything V2 weights into a LightlyTrain checkpoint.

    Args:
        out:
            Destination path for the exported LightlyTrain ``.pt`` checkpoint.
        model_name:
            The Depth Anything V2 model name to convert.
        weights:
            Optional path to a local official checkpoint (``.pth``/``.safetensors``).
            When omitted, the official weights are downloaded from Hugging Face.

    Returns:
        The path to the exported checkpoint.
    """
    parsed_name = _parse_model_name(model_name)
    license_info = _MODEL_LICENSES[parsed_name]
    is_apache = license_info == "Apache-2.0"
    if not is_apache:
        logger.warning(
            f"'{parsed_name}' is licensed under {license_info}, NOT Apache-2.0. "
            "LightlyTrain does not host or redistribute it; you are responsible for "
            "complying with its license terms."
        )

    model_cls = _MODEL_CLASSES[parsed_name]
    model = model_cls(
        model_name=parsed_name,
        load_weights=False,
    )

    if weights is None:
        hf_weights = _HF_WEIGHTS[parsed_name]
        if not is_apache:
            raise ValueError(
                f"Refusing to download '{parsed_name}' from Hugging Face because it is "
                f"licensed under {license_info} (not Apache-2.0). Download "
                f"'{hf_weights['filename']}' from '{hf_weights['repo_id']}' yourself "
                "and pass it via `--weights <path>`."
            )
        logger.info(
            f"Downloading official weights from '{hf_weights['repo_id']}' "
            f"({hf_weights['filename']})..."
        )
        source_path = _download_huggingface_weights(
            repo_id=hf_weights["repo_id"],
            filename=hf_weights["filename"],
        )
    else:
        source_path = Path(weights).expanduser()

    logger.info(f"Loading official weights from '{source_path}'...")
    _load_official_weights(model=model, path=source_path)

    model_dict = {
        "model_class_path": model.class_path,
        "model_init_args": model.init_args,
        "train_model": model.state_dict(),
        "license_info": license_info,
    }

    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_dict, out)

    checkpoint_hash = task_model_helpers.checkpoint_hash(out)
    logger.info(f"Exported LightlyTrain checkpoint to '{out}'.")
    logger.info(f"sha256: {checkpoint_hash}")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination path for the exported LightlyTrain .pt checkpoint.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov2/dav2-relative-large",
        help="The Depth Anything V2 model name to convert.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Optional path to a local official checkpoint. When omitted, the official "
            "weights are downloaded from Hugging Face."
        ),
    )
    args = parser.parse_args()

    convert_checkpoint(
        out=args.out,
        model_name=args.model_name,
        weights=args.weights,
    )


def _parse_model_name(model_name: str) -> str:
    key = model_name.lower()
    if key in _MODEL_CLASSES:
        return key
    raise ValueError(
        f"Model name '{model_name}' is not supported. Available models are: "
        f"{sorted(_MODEL_CLASSES)}."
    )


def _download_huggingface_weights(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Loading Depth Anything V2 weights from Hugging Face requires "
            "'huggingface-hub'. Install it or pass a local weights path."
        ) from error

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _load_official_weights(
    model: DepthAnythingDepthEstimation,
    path: Path,
) -> None:
    state_dict = _load_state_dict_file(path)
    remapped = _remap_official_da2_keys(state_dict)
    incompatible = model.load_state_dict(remapped, strict=False)

    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if not missing <= _ALLOWED_MISSING_OFFICIAL_KEYS or unexpected:
        raise RuntimeError(
            "Could not load Depth Anything V2 state dict. "
            f"Missing keys: {sorted(missing - _ALLOWED_MISSING_OFFICIAL_KEYS)}; "
            f"unexpected keys: {sorted(unexpected)}"
        )


def _load_state_dict_file(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise ValueError(f"Checkpoint file '{path}' does not exist.")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                "Loading '.safetensors' Depth Anything V2 weights requires "
                "'safetensors'. Install it or pass a PyTorch checkpoint."
            ) from error
        return load_file(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "train_model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        return dict(checkpoint)

    raise ValueError(f"Unsupported checkpoint format in '{path}'.")


def _remap_official_da2_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remap Depth Anything V2 checkpoint keys onto the flat TaskModel layout.

    The official checkpoint stores the backbone under ``pretrained.*`` and the DPT
    decoder under ``depth_head.*``. This maps them onto the TaskModel layout:
    ``backbone.*`` and ``decoder.*``. The unused ``pretrained.mask_token`` is dropped
    (see ``_ALLOWED_MISSING_OFFICIAL_KEYS``), as are the dead
    ``depth_head.scratch.refinenet4.resConfUnit1.*`` weights: refinenet4 receives a
    single input, so its resConfUnit1 never runs in forward. The model omits the module
    entirely, so dropping these keys keeps the load strict (no unexpected keys).
    """
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.endswith("mask_token"):
            continue
        if ".scratch.refinenet4.resConfUnit1." in key:
            continue
        new_key = key
        if new_key.startswith("pretrained."):
            new_key = "backbone." + new_key[len("pretrained.") :]
        elif new_key.startswith("depth_head."):
            new_key = "decoder." + new_key[len("depth_head.") :]
        remapped[new_key] = value
    return remapped


if __name__ == "__main__":
    main()
