#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers import (
    MemEffAttention,
    NestedTensorBlock,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.depth_anything_v3_depth_estimation.dpt_head import (
    DPTHead,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

_BACKBONE_SPECS: dict[str, dict[str, int]] = {
    "vits": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vitb": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vitg": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
}

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "da3mono-large": {
        "canonical_name": "depth-anything-v3/da3mono-large",
        "hf_repo_id": "depth-anything/DA3MONO-LARGE",
        "hf_filename": "model.safetensors",
        "model_args": {
            "backbone_name": "vitl",
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "dim_in": 1024,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
            "use_sky_head": True,
        },
    }
}

_MODEL_ALIASES: dict[str, str] = {
    "da3mono-large": "da3mono-large",
    "depth-anything-v3/da3mono-large": "da3mono-large",
    "depth-anything/da3mono-large": "da3mono-large",
}

_DEFAULT_IMAGE_NORMALIZE = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

_ALLOWED_MISSING_OFFICIAL_KEYS = {"backbone.mask_token"}


class DepthAnythingV3MonocularDepthEstimation(TaskModel):
    """Depth Anything V3 monocular relative-depth inference model."""

    model_suffix = "depth-estimation"

    def __init__(
        self,
        *,
        model_name: str = "depth-anything-v3/da3mono-large",
        image_size: int = 504,
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        weights: PathLike | None = None,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """
        Args:
            model_name:
                The Depth Anything V3 model name. Supported names are
                ``"depth-anything-v3/da3mono-large"``, ``"depth-anything/DA3MONO-LARGE"``,
                and ``"da3mono-large"``.
            image_size:
                Upper bound for the longest image side during inference. The resized
                height and width are rounded to the nearest multiple of the DA3 patch
                size. The official DA3 inference default is 504.
            image_normalize:
                Image normalization parameters. Defaults to ImageNet statistics.
            weights:
                Optional path to a local ``.pt``/``.pth``/``.safetensors`` checkpoint.
                If omitted and ``load_weights=True``, weights are downloaded from the
                official Hugging Face repository.
            model_args:
                Additional arguments controlling backbone and DPT head construction.
            load_weights:
                If False, no pretrained weights are loaded.
        """
        super().__init__(locals(), ignore_args={"weights", "load_weights"})
        parsed_name = self.parse_model_name(model_name)
        config = _MODEL_CONFIGS[parsed_name]

        self.model_name = config["canonical_name"]
        self.image_size = image_size
        self.image_normalize = (
            _DEFAULT_IMAGE_NORMALIZE if image_normalize is None else image_normalize
        )

        net_args = dict(config["model_args"])
        if model_args is not None:
            net_args.update(model_args)

        backbone_name: str = net_args["backbone_name"]
        backbone_spec = _resolve_backbone_spec(
            backbone_name=backbone_name,
            backbone_args=net_args.get("backbone_args"),
        )
        patch_size = int(net_args.get("patch_size", 14))
        dim_in = int(net_args.get("dim_in", backbone_spec["embed_dim"]))

        self.out_layers: tuple[int, ...] = tuple(net_args["out_layers"])
        self.patch_size = patch_size
        self.backbone = _make_dinov2_backbone(
            image_size=int(net_args["image_size"]),
            patch_size=patch_size,
            **backbone_spec,
        )
        self.head = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=int(net_args.get("output_dim", 1)),
            features=int(net_args.get("features", 256)),
            out_channels=tuple(net_args.get("out_channels", (256, 512, 1024, 1024))),
            use_sky_head=bool(net_args.get("use_sky_head", True)),
        )

        if load_weights:
            weights_path = (
                _download_huggingface_weights(
                    repo_id=config["hf_repo_id"],
                    filename=config["hf_filename"],
                )
                if weights is None
                else Path(weights).expanduser()
            )
            self.load_weights(weights_path)

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [config["canonical_name"] for config in _MODEL_CONFIGS.values()]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            cls.parse_model_name(model_name=model)
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def parse_model_name(cls, model_name: str) -> str:
        key = model_name.lower()
        try:
            return _MODEL_ALIASES[key]
        except KeyError:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available models are: "
                f"{cls.list_model_names()}."
            )

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> Tensor:
        """Returns a monocular relative-depth map for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape ``(C, H, W)``.

        Returns:
            A depth tensor of shape ``(H, W)`` at the original input resolution.
            Larger values correspond to farther scene content.
        """
        self._track_inference()
        if self.training:
            self.eval()

        x = file_helpers.as_image_tensor(image)
        image_h, image_w = x.shape[-2:]
        x = self._preprocess_image(x)

        out = self.forward(x.unsqueeze(0))
        depth = F.interpolate(
            out["depth"],
            size=(image_h, image_w),
            mode="bilinear",
            align_corners=False,
        )
        return depth[0, 0]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run depth inference on a preprocessed batch with shape ``(B, 3, H, W)``."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        feats = self._extract_features(x)
        out = self.head(feats=feats, H=x.shape[-2], W=x.shape[-1])
        _set_mono_sky_regions_to_max_depth(out)
        return out

    def load_weights(self, path: PathLike) -> None:
        state_dict = _load_state_dict_file(Path(path))
        self._load_depth_anything_state_dict(state_dict)

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        self._load_depth_anything_state_dict(state_dict)

    def _extract_features(self, x: Tensor) -> list[Tensor]:
        intermediate = self.backbone.get_intermediate_layers(
            x,
            n=self.out_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        return [patch_tokens for patch_tokens, _class_token in intermediate]

    def _preprocess_image(self, image: Tensor) -> Tensor:
        first_param = next(self.parameters())
        device = first_param.device
        dtype = first_param.dtype

        image = _ensure_three_channel_image(image).to(device)
        image = transforms_functional.to_dtype(image, dtype=dtype, scale=True)
        image = _resize_longest_side_to_upper_bound(
            image=image,
            upper_bound=self.image_size,
        )
        image = _resize_to_nearest_patch_multiple(
            image=image,
            patch_size=self.patch_size,
        )
        return transforms_functional.normalize(
            image,
            mean=self.image_normalize["mean"],
            std=self.image_normalize["std"],
        )

    def _load_depth_anything_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        remapped = _remap_official_da3_keys(state_dict)
        incompatible = self.load_state_dict(remapped, strict=False)

        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if not missing <= _ALLOWED_MISSING_OFFICIAL_KEYS or unexpected:
            raise RuntimeError(
                "Could not load Depth Anything V3 state dict. "
                f"Missing keys: {sorted(missing - _ALLOWED_MISSING_OFFICIAL_KEYS)}; "
                f"unexpected keys: {sorted(unexpected)}"
            )


def _make_dinov2_backbone(
    *,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=0,
        init_values=1.0,
        block_chunks=0,
    )


def _resolve_backbone_spec(
    backbone_name: str, backbone_args: Mapping[str, Any] | None
) -> dict[str, int]:
    spec = dict(_BACKBONE_SPECS.get(backbone_name, {}))
    if backbone_args is not None:
        spec.update(backbone_args)
    if not {"embed_dim", "depth", "num_heads"}.issubset(spec):
        raise ValueError(
            "backbone_args must define 'embed_dim', 'depth', and 'num_heads' "
            "when using a custom DA3 backbone."
        )
    return {key: int(spec[key]) for key in ("embed_dim", "depth", "num_heads")}


def _set_mono_sky_regions_to_max_depth(out: dict[str, Tensor]) -> None:
    if "depth" not in out or "sky" not in out:
        return

    non_sky_mask = out["sky"] < 0.3
    if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
        return

    non_sky_depth = out["depth"][non_sky_mask]
    if non_sky_depth.numel() > 100_000:
        idx = torch.randint(
            0,
            non_sky_depth.numel(),
            (100_000,),
            device=non_sky_depth.device,
        )
        non_sky_depth = non_sky_depth[idx]
    non_sky_max = torch.quantile(non_sky_depth, 0.99)

    depth = out["depth"].clone()
    depth[~non_sky_mask] = non_sky_max
    out["depth"] = depth


def _download_huggingface_weights(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Loading Depth Anything V3 weights from Hugging Face requires "
            "'huggingface-hub'. Install it or pass a local weights path."
        ) from error

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


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

    Handles three input layouts:
    - Flat (already matches the model): ``backbone.<rest>``, ``head.<rest>``
    - Lightning training: ``model.backbone.<rest>``, ``model.head.<rest>``
    - HF official: ``model.backbone.pretrained.<rest>``, ``model.head.<rest>``
    """
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        if new_key.startswith("backbone.pretrained."):
            new_key = "backbone." + new_key[len("backbone.pretrained.") :]
        remapped[new_key] = value
    return remapped


def _ensure_three_channel_image(image: Tensor) -> Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(image.shape)}.")

    channels = image.shape[0]
    if channels == 1:
        return image.expand(3, -1, -1)
    if channels == 3:
        return image
    if channels >= 4:
        return image[:3]
    raise ValueError(f"Expected 1, 3, or 4 image channels, got {channels}.")


def _resize_longest_side_to_upper_bound(image: Tensor, upper_bound: int) -> Tensor:
    h, w = image.shape[-2:]
    longest = max(h, w)
    if longest == upper_bound:
        return image

    scale = upper_bound / float(longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return transforms_functional.resize(image, size=[new_h, new_w])


def _resize_to_nearest_patch_multiple(image: Tensor, patch_size: int) -> Tensor:
    h, w = image.shape[-2:]
    new_h = max(patch_size, _nearest_multiple(h, patch_size))
    new_w = max(patch_size, _nearest_multiple(w, patch_size))
    if (new_h, new_w) == (h, w):
        return image
    return transforms_functional.resize(image, size=[new_h, new_w])


def _nearest_multiple(value: int, multiple: int) -> int:
    down = (value // multiple) * multiple
    up = down + multiple
    return up if abs(up - value) <= abs(value - down) else down
