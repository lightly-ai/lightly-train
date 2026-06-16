#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._data import cache, download, file_helpers
from lightly_train._env import Env
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.depth_estimation_components import image_utils
from lightly_train._task_models.depth_estimation_components.dpt import DPT
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2/dav2-relative-small": {
        "canonical_name": "dinov2/dav2-relative-small",
        "backbone_name": "vits14-noreg",
        # TODO(Nauryzbay, 06/2026): Host the converted checkpoint and set its URL so
        # `load_weights=True` can download it. Until then pass a local `weights` path.
        "weights_url": None,
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 64,
            "out_channels": (48, 96, 192, 384),
            "output_dim": 1,
        },
    },
    "dinov2/dav2-relative-base": {
        "canonical_name": "dinov2/dav2-relative-base",
        "backbone_name": "vitb14-noreg",
        "weights_url": None,
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 128,
            "out_channels": (96, 192, 384, 768),
            "output_dim": 1,
        },
    },
    "dinov2/dav2-relative-large": {
        "canonical_name": "dinov2/dav2-relative-large",
        "backbone_name": "vitl14-noreg",
        "weights_url": None,
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
        },
    },
}


class DepthAnythingV2RelativeDepthEstimation(TaskModel):
    """Depth Anything V2 relative-depth inference model."""

    model_suffix = "dav2_relative"

    def __init__(
        self,
        *,
        model_name: str = "dinov2/dav2-relative-large",
        process_resolution: int = 518,
        model_args: dict[str, Any] | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
        weights: PathLike | None = None,
    ) -> None:
        """
        Args:
            model_name:
                The Depth Anything V2 model name. One of
                ``"dinov2/dav2-relative-small"``, ``"dinov2/dav2-relative-base"``, or
                ``"dinov2/dav2-relative-large"``.
            process_resolution:
                Target size for the shorter image side during inference. The resized
                height and width are rounded to a multiple of the patch size. The
                official Depth Anything V2 inference default is 518.
            model_args:
                Additional arguments controlling the DPT decoder and feature
                extraction, e.g. ``out_layers``, ``features``, ``out_channels``, or
                ``output_dim``.
            backbone_args:
                Additional arguments passed to the DINOv2 backbone construction (see
                ``DINOV2_VIT_PACKAGE.get_model``), e.g. ``in_chans`` or
                ``drop_path_rate``. These override the Depth Anything V2 defaults.
            load_weights:
                If True, load the converted Depth Anything V2 weights (backbone and DPT
                head) so the model is a ready-to-use depth predictor; a local
                ``weights`` path takes precedence over the hosted URL. If False, the
                model is randomly initialized, e.g. when restoring from an exported
                checkpoint via ``load_train_state_dict``.
            weights:
                Optional path to a converted Depth Anything V2 checkpoint (the
                ``convert_checkpoint_dav2`` output) to load instead of the hosted
                weights. Intended for debugging before the checkpoint is hosted.
        """
        super().__init__(locals(), ignore_args={"load_weights", "weights"})
        key = model_name.lower()
        if key not in _MODEL_CONFIGS:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available models are: "
                f"{self.list_model_names()}."
            )
        config = _MODEL_CONFIGS[key]

        self.model_name = config["canonical_name"]
        self.process_resolution = process_resolution

        net_args = dict(config["model_args"])
        if model_args is not None:
            net_args.update(model_args)

        patch_size = int(net_args["patch_size"])
        self.out_layers: tuple[int, ...] = tuple(net_args["out_layers"])
        self.patch_size = patch_size

        # Reproduce the plain (register-free, unchunked, MLP-FFN) ViT that Depth Anything
        # V2 is built on so the state dict keys match the checkpoint; `block_chunks=0`
        # keeps the `blocks.{i}.` key layout. The backbone is built without weights: when
        # `load_weights` is True the converted DA2 checkpoint is loaded below instead.
        backbone_model_args: dict[str, Any] = {
            "img_size": int(net_args["image_size"]),
            "ffn_layer": "mlp",
            "block_chunks": 0,
            "drop_path_rate": 0.0,
            "init_values": 1.0,
            "num_register_tokens": 0,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
        }
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)
        self.backbone = DINOV2_VIT_PACKAGE.get_model(
            model_name=config["backbone_name"],
            model_args=backbone_model_args,
            load_weights=False,
        )
        self.decoder = DPT(
            dim_in=int(self.backbone.embed_dim),
            patch_size=patch_size,
            output_dim=int(net_args["output_dim"]),
            features=int(net_args["features"]),
            out_channels=tuple(net_args["out_channels"]),
            activation="relu",
            use_sky_head=False,
        )

        if load_weights:
            _load_pretrained_weights(
                model=self,
                weights_url=config["weights_url"],
                weights=weights,
            )

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [config["canonical_name"] for config in _MODEL_CONFIGS.values()]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        return model.lower() in _MODEL_CONFIGS

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> Tensor:
        """Returns a relative-depth map for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape ``(C, H, W)``; uint8 tensors are interpreted in [0, 255] and
                float tensors in [0, 1].

        Returns:
            A depth tensor of shape ``(H, W)`` matching the original input resolution.
            Larger values correspond to nearer scene content.
        """
        self._track_inference()
        if self.training:
            self.eval()

        x, metadata = self.preprocess_image(image)
        batch = self.preprocess_batch([x])
        raw = self.forward(batch)
        return self.postprocess(raw, [metadata])[0]

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
    ) -> list[Tensor]:
        """Returns relative-depth maps for the given batch of images.

        Args:
            images:
                Sequence of input images. Each can be a path, URL, PIL image, or
                tensor. Tensors must have shape ``(C, H, W)``; uint8 tensors are
                interpreted in [0, 255] and float tensors in [0, 1].

        Returns:
            One depth tensor of shape ``(H, W)`` per image, matching each image's
            original resolution. Larger values correspond to nearer scene content.
            Images whose processed sizes differ are center-cropped to the smallest size
            in the batch before inference, so their depth maps are slightly stretched
            when resized back.
        """
        self._track_inference()
        if self.training:
            self.eval()

        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for image in images:
            x, meta = self.preprocess_image(image)
            tensors.append(x)
            metadata.append(meta)
        batch = self.preprocess_batch(tensors)
        raw = self.forward(batch)
        return self.postprocess(raw, metadata)

    def preprocess_image(
        self, image: PathLike | PILImage | Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        """Per-image preprocessing producing a model-input tensor and metadata.

        The aspect-preserving resize means outputs across a batch may have different
        shapes, so they are not always stackable; `preprocess_batch` therefore takes a
        sequence and unifies the sizes.
        """
        x = file_helpers.as_image_tensor(image)
        image_h, image_w = x.shape[-2:]
        x = image_utils.process_image_dav2(
            x, process_resolution=self.process_resolution
        )
        device = next(self.parameters()).device
        return x.to(device=device), {"orig_h": image_h, "orig_w": image_w}

    def preprocess_batch(  # type: ignore[override]
        self, batch: Sequence[Tensor]
    ) -> Tensor:
        stacked = image_utils.process_batch(batch)
        return stacked.to(dtype=next(self.parameters()).dtype)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run depth inference on a preprocessed batch with shape ``(B, 3, H, W)``."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        feats = self._extract_features(x)
        out: dict[str, Tensor] = self.decoder(feats=feats, H=x.shape[-2], W=x.shape[-1])
        return out

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: dict[str, Tensor],
        metadata: Sequence[dict[str, Any]],
    ) -> list[Tensor]:
        """Maps raw forward outputs to one depth tensor per image, bilinearly resized
        to the original input size (``orig_h``, ``orig_w`` from the metadata)."""
        depth_batch = raw_outputs["depth"]
        out: list[Tensor] = []
        for i, meta in enumerate(metadata):
            depth = depth_batch[i, 0]
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            if depth.shape != (orig_h, orig_w):
                depth = F.interpolate(
                    depth[None, None],
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]
            out.append(depth)
        return out

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.load_state_dict(state_dict, strict=True)

    def _extract_features(self, x: Tensor) -> list[Tensor]:
        intermediate = self.backbone.get_intermediate_layers(
            x,
            n=self.out_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        return [patch_tokens for patch_tokens, _class_token in intermediate]


def _load_pretrained_weights(
    model: DepthAnythingV2RelativeDepthEstimation,
    *,
    weights_url: str | None,
    weights: PathLike | None,
) -> None:
    """Loads the converted Depth Anything V2 checkpoint into the model in place.

    A local ``weights`` path takes precedence; otherwise the checkpoint is downloaded
    from ``weights_url`` into the model cache. Both come from ``convert_checkpoint_dav2``.
    """
    if weights is not None:
        checkpoint_path = Path(weights).expanduser()
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file '{checkpoint_path}' does not exist.")
    elif weights_url is not None:
        checkpoint_path = cache.get_model_cache_dir() / Path(weights_url).name
        if not checkpoint_path.exists():
            logger.info(
                f"Downloading Depth Anything V2 weights from '{weights_url}' to "
                f"'{checkpoint_path}'."
            )
            download.download_from_url(
                weights_url,
                checkpoint_path,
                timeout=Env.LIGHTLY_TRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC.value,
            )
        else:
            logger.info(
                f"Using cached Depth Anything V2 weights from '{checkpoint_path}'."
            )
    else:
        raise RuntimeError(
            "No pretrained Depth Anything V2 checkpoint is available yet: the hosted "
            "weights URL is not set. Pass `weights=<converted .pt>` (produced by the "
            "convert_checkpoint_dav2 script) to load a local checkpoint, or set "
            "`load_weights=False`."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping) and "train_model" in checkpoint:
        state_dict = dict(checkpoint["train_model"])
    else:
        state_dict = dict(checkpoint)
    model.load_train_state_dict(state_dict)
