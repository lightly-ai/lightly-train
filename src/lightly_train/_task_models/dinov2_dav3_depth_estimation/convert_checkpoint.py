#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Convert official Depth Anything V3 weights into a LightlyTrain checkpoint.

This is a one-off conversion utility and the single place that understands the
official Depth Anything V3 weight format. It downloads the official Hugging Face
weights, remaps their keys onto the flat ``TaskModel`` layout, loads them into
``DepthAnythingV3MonocularDepthEstimation``, and re-exports them in the LightlyTrain
``.pt`` format (``model_class_path`` + ``model_init_args`` + ``train_model`` state dict)
so the model can afterwards be loaded with ``lightly_train.load_model(<out>)`` without
ever touching Hugging Face again.

Example:
    python -m lightly_train._task_models.dinov2_dav3_depth_estimation.convert_checkpoint \\
        --out da3mono_large.pt
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
from lightly_train._task_models.dinov2_dav3_depth_estimation.task_model import (
    DepthAnythingV3MonocularDepthEstimation,
)

logger = logging.getLogger(__name__)

# Official Hugging Face weights per (parsed) model name.
_HF_WEIGHTS: dict[str, dict[str, str]] = {
    "da3mono-large": {
        "repo_id": "depth-anything/DA3MONO-LARGE",
        "filename": "model.safetensors",
    },
}

# ``backbone.mask_token`` only exists for masked-image-modeling pretraining and is
# absent from the official inference checkpoints. It is never read during depth
# inference, so we tolerate it being missing and keep the model's random init.
_ALLOWED_MISSING_OFFICIAL_KEYS = {"backbone.mask_token"}


def convert_checkpoint(
    out: Path,
    model_name: str = "depth-anything-v3/da3mono-large",
    weights: Path | None = None,
) -> Path:
    """Convert official Depth Anything V3 weights into a LightlyTrain checkpoint.

    Args:
        out:
            Destination path for the exported LightlyTrain ``.pt`` checkpoint.
        model_name:
            The Depth Anything V3 model name to convert.
        weights:
            Optional path to a local official checkpoint (``.safetensors``/``.pt``).
            When omitted, the official weights are downloaded from Hugging Face.

    Returns:
        The path to the exported checkpoint.
    """
    model = DepthAnythingV3MonocularDepthEstimation(
        model_name=model_name,
        load_weights=False,
    )

    if weights is None:
        parsed_name = DepthAnythingV3MonocularDepthEstimation.parse_model_name(
            model_name
        )
        hf_weights = _HF_WEIGHTS[parsed_name]
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
        "license_info": "",
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
        default="depth-anything-v3/da3mono-large",
        help="The Depth Anything V3 model name to convert.",
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


def _download_huggingface_weights(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Loading Depth Anything V3 weights from Hugging Face requires "
            "'huggingface-hub'. Install it or pass a local weights path."
        ) from error

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _load_official_weights(
    model: DepthAnythingV3MonocularDepthEstimation, path: Path
) -> None:
    state_dict = _load_state_dict_file(path)
    remapped = _remap_official_da3_keys(state_dict)
    incompatible = model.load_state_dict(remapped, strict=False)

    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if not missing <= _ALLOWED_MISSING_OFFICIAL_KEYS or unexpected:
        raise RuntimeError(
            "Could not load Depth Anything V3 state dict. "
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
                "Loading '.safetensors' Depth Anything V3 weights requires "
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


def _remap_official_da3_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remap Depth Anything V3 checkpoint keys onto the flat TaskModel layout.

    The official checkpoint stores the backbone under ``backbone.pretrained.*`` and the
    DPT decoder under ``head.*`` (optionally behind a ``model.`` prefix). This maps them
    onto the TaskModel layout: ``backbone.*`` and ``decoder.*``.
    """
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key.startswith("backbone.pretrained."):
            new_key = "backbone." + new_key[len("backbone.pretrained.") :]
        elif new_key.startswith("head."):
            new_key = "decoder." + new_key[len("head.") :]
        remapped[new_key] = value
    return remapped


if __name__ == "__main__":
    main()
