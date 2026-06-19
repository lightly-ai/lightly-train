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
from typing import Any

import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._data import file_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.depth_estimation_components import image_utils
from lightly_train._task_models.depth_estimation_components.dpt import DPT
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

# The official Depth Anything V2 metric models are trained per domain with a fixed
# maximum depth that the sigmoid head output is scaled by: 20 m for the indoor Hypersim
# models and 80 m for the outdoor Virtual KITTI 2 models. The per-size backbone and DPT
# arguments are identical to the relative models.
_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2/dav2-metric-small-hypersim": {
        "canonical_name": "dinov2/dav2-metric-small-hypersim",
        "backbone_name": "vits14-noreg",
        "inference_size": 518,
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 64,
            "out_channels": (48, 96, 192, 384),
            "output_dim": 1,
            "max_depth": 20.0,
        },
    },
    "dinov2/dav2-metric-base-hypersim": {
        "canonical_name": "dinov2/dav2-metric-base-hypersim",
        "backbone_name": "vitb14-noreg",
        "inference_size": 518,
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 128,
            "out_channels": (96, 192, 384, 768),
            "output_dim": 1,
            "max_depth": 20.0,
        },
    },
    "dinov2/dav2-metric-large-hypersim": {
        "canonical_name": "dinov2/dav2-metric-large-hypersim",
        "backbone_name": "vitl14-noreg",
        "inference_size": 518,
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
            "max_depth": 20.0,
        },
    },
    "dinov2/dav2-metric-small-vkitti": {
        "canonical_name": "dinov2/dav2-metric-small-vkitti",
        "backbone_name": "vits14-noreg",
        "inference_size": 518,
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 64,
            "out_channels": (48, 96, 192, 384),
            "output_dim": 1,
            "max_depth": 80.0,
        },
    },
    "dinov2/dav2-metric-base-vkitti": {
        "canonical_name": "dinov2/dav2-metric-base-vkitti",
        "backbone_name": "vitb14-noreg",
        "inference_size": 518,
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 128,
            "out_channels": (96, 192, 384, 768),
            "output_dim": 1,
            "max_depth": 80.0,
        },
    },
    "dinov2/dav2-metric-large-vkitti": {
        "canonical_name": "dinov2/dav2-metric-large-vkitti",
        "backbone_name": "vitl14-noreg",
        "inference_size": 518,
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
            "max_depth": 80.0,
        },
    },
}


class DepthAnythingV2MetricDepthEstimation(TaskModel):
    """Depth Anything V2 metric-depth inference model."""

    model_suffix = "dav2_metric"

    def __init__(
        self,
        *,
        model_name: str = "dinov2/dav2-metric-large-hypersim",
        model_args: dict[str, Any] | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
        weights: PathLike | None = None,
    ) -> None:
        """
        Args:
            model_name:
                The Depth Anything V2 metric model name. One of
                ``"dinov2/dav2-metric-{small,base,large}-hypersim"`` (indoor, 20 m) or
                ``"dinov2/dav2-metric-{small,base,large}-vkitti"`` (outdoor, 80 m).
            model_args:
                Additional arguments controlling the DPT decoder and feature
                extraction, e.g. ``out_layers``, ``features``, ``out_channels``,
                ``output_dim``, or ``max_depth``.
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
                Optional path to a converted Depth Anything V2 metric checkpoint (the
                ``convert_checkpoint_dav2`` output) to load instead of the hosted
                weights. Required for the non-Apache-2.0 models, which are not hosted.
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
        # The inference size is fixed per model: it is the official Depth Anything V2
        # inference resolution and predictions are resized back to the original image
        # size, so it is not a user-facing parameter.
        self.inference_size = int(config["inference_size"])

        net_args = dict(config["model_args"])
        if model_args is not None:
            net_args.update(model_args)

        patch_size = int(net_args["patch_size"])
        self.out_layers: tuple[int, ...] = tuple(net_args["out_layers"])
        self.patch_size = patch_size
        # Fixed maximum depth in meters that the sigmoid head output is scaled by.
        self.max_depth = float(net_args["max_depth"])

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
        # The metric DPT head ends in a sigmoid (vs the relative head's relu); its raw
        # output in [0, 1] is scaled by `max_depth` in `postprocess`.
        self.decoder = DPT(
            dim_in=int(self.backbone.embed_dim),
            patch_size=patch_size,
            output_dim=int(net_args["output_dim"]),
            features=int(net_args["features"]),
            out_channels=tuple(net_args["out_channels"]),
            activation="sigmoid",
            use_sky_head=False,
        )

        if load_weights:
            _load_pretrained_weights(
                model=self,
                weights=weights,
                canonical_name=config["canonical_name"],
            )

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [config["canonical_name"] for config in _MODEL_CONFIGS.values()]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        return model.lower() in _MODEL_CONFIGS

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> Tensor:
        """Returns a metric-depth map for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape ``(C, H, W)``; uint8 tensors are interpreted in [0, 255] and
                float tensors in [0, 1].

        Returns:
            A depth tensor of shape ``(H, W)`` matching the original input resolution.
            Values are metric depth in meters; larger values correspond to farther
            scene content.
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
        """Returns metric-depth maps for the given batch of images.

        Args:
            images:
                Sequence of input images. Each can be a path, URL, PIL image, or
                tensor. Tensors must have shape ``(C, H, W)``; uint8 tensors are
                interpreted in [0, 255] and float tensors in [0, 1].

        Returns:
            One depth tensor of shape ``(H, W)`` per image, matching each image's
            original resolution. Values are metric depth in meters; larger values
            correspond to farther scene content. Images whose processed sizes differ are
            center-cropped to the smallest size in the batch before inference, so their
            depth maps are slightly stretched when resized back.
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
        x = image_utils.process_image_dav2(x, process_res=self.inference_size)
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
        """Maps raw forward outputs to one metric-depth tensor per image.

        The sigmoid head output in [0, 1] is scaled by ``max_depth`` at the processing
        resolution (matching the official ``head(...) * max_depth`` order) before being
        bilinearly resized to the original input size (``orig_h``, ``orig_w`` from the
        metadata)."""
        depth_batch = raw_outputs["depth"]
        out: list[Tensor] = []
        for i, meta in enumerate(metadata):
            depth = depth_batch[i, 0]
            # Scale the sigmoid output to metric depth before the resize, matching the
            # official metric model's `head(...) * max_depth` order.
            depth = depth * self.max_depth
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
    model: DepthAnythingV2MetricDepthEstimation,
    *,
    weights: PathLike | None,
    canonical_name: str,
) -> None:
    """Loads the converted Depth Anything V2 metric checkpoint into the model in place.

    A local converted ``weights`` path takes precedence. Otherwise the checkpoint is
    resolved via ``task_model_helpers.download_checkpoint`` (downloaded to the model
    cache and verified against its sha256) -- but only the Apache-2.0 models are hosted.
    The non-commercial models are not redistributed: convert them locally with
    ``convert_checkpoint_dav2`` and pass the result via ``weights``.
    """
    if weights is not None:
        checkpoint: PathLike = weights
    elif canonical_name in task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH:
        checkpoint = canonical_name
    else:
        raise RuntimeError(
            f"'{canonical_name}' is not hosted by LightlyTrain. Make sure you "
            "understand its license and how it applies to your use, then convert the "
            "weights locally:\n"
            "  1. Download the official Depth Anything V2 checkpoint from Hugging "
            "Face.\n"
            "  2. Convert it (run without `--weights` to see which file to download):\n"
            "       python -m lightly_train._task_models.depth_estimation_components"
            ".convert_checkpoint_dav2 --model-name "
            f"{canonical_name} --weights <official.pth> --out <converted.pt>\n"
            "  3. Reload with `weights=<converted.pt>`, or set `load_weights=False` "
            "to skip."
        )

    checkpoint_path = task_model_helpers.download_checkpoint(checkpoint=checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "train_model" in state:
        state_dict = dict(state["train_model"])
    else:
        state_dict = dict(state)
    model.load_train_state_dict(state_dict)
