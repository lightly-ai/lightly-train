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
from lightly_train._task_models.depth_estimation_components import (
    image_utils as depth_image_utils,
)
from lightly_train._task_models.depth_estimation_components.dpt import DPT
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2/dav3-relative-large": {
        "canonical_name": "dinov2/dav3-relative-large",
        "backbone_name": "vitl14-noreg",
        "inference_size": 504,
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
            "use_sky_head": True,
        },
    }
}


class DepthAnythingV3RelativeDepthEstimation(TaskModel):
    """Depth Anything V3 relative-depth inference model."""

    model_suffix = "dav3_relative"

    def __init__(
        self,
        *,
        model_name: str = "dinov2/dav3-relative-large",
        model_args: dict[str, Any] | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
        weights: PathLike | None = None,
    ) -> None:
        """
        Args:
            model_name:
                The Depth Anything V3 model name. The only supported name is
                ``"dinov2/dav3-relative-large"``.
            model_args:
                Additional arguments controlling the DPT decoder and feature
                extraction, e.g. ``out_layers``, ``features``, ``out_channels``,
                ``output_dim``, or ``use_sky_head``.
            backbone_args:
                Additional arguments passed to the DINOv2 backbone construction (see
                ``DINOV2_VIT_PACKAGE.get_model``), e.g. ``in_chans`` or
                ``drop_path_rate``. These override the Depth Anything V3 defaults.
            load_weights:
                If True, load the converted Depth Anything V3 weights (backbone and DPT
                head) so the model is a ready-to-use depth predictor; a local
                ``weights`` path takes precedence over the hosted URL. If False, the
                model is randomly initialized, e.g. when restoring from an exported
                checkpoint via ``load_train_state_dict``.
            weights:
                Optional path to a converted Depth Anything V3 checkpoint (the
                ``convert_checkpoint_dav3`` output) to load instead of the hosted
                weights.
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
        # The inference size is fixed per model: Depth Anything V3 was trained at this
        # resolution and predictions are resized back to the original image size, so it
        # is not a user-facing parameter.
        self.inference_size = int(config["inference_size"])

        self.process_res_method = "upper_bound_resize"

        net_args = dict(config["model_args"])
        if model_args is not None:
            net_args.update(model_args)

        patch_size = int(net_args["patch_size"])
        self.out_layers: tuple[int, ...] = tuple(net_args["out_layers"])
        self.patch_size = patch_size

        # Reproduce the plain (register-free, unchunked, MLP-FFN) ViT-L that DA3 is
        # built on so the state dict keys match the checkpoint; `block_chunks=0` keeps
        # the `blocks.{i}.` key layout. The backbone is built without weights: when
        # `load_weights` is True the converted DA3 checkpoint is loaded below instead.
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
            use_sky_head=bool(net_args["use_sky_head"]),
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
        """Returns a relative-depth map for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape ``(C, H, W)``; uint8 tensors are interpreted in [0, 255] and
                float tensors in [0, 1].

        Returns:
            A depth tensor of shape ``(H, W)`` matching the original input resolution.
            Larger values correspond to farther scene content.
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
            original resolution. Images whose processed sizes differ are
            center-cropped to the smallest size in the batch before inference, so
            their depth maps are slightly stretched when resized back.
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
        # Process on the input's native device: the cv2-parity resize rounds after an
        # einsum whose accumulation order differs between CPU and GPU, so moving to the
        # model device first could flip pixels and break bit-exactness.
        x = depth_image_utils.process_image_dav3(
            x,
            process_res=self.inference_size,
            process_res_method=self.process_res_method,
        )
        device = next(self.parameters()).device
        return x.to(device=device), {"orig_h": image_h, "orig_w": image_w}

    def preprocess_batch(  # type: ignore[override]
        self, batch: Sequence[Tensor]
    ) -> Tensor:
        stacked = depth_image_utils.process_batch(batch)
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
        sky_batch = raw_outputs.get("sky")
        out: list[Tensor] = []
        for i, meta in enumerate(metadata):
            depth = depth_batch[i, 0]
            sky = None if sky_batch is None else sky_batch[i, 0]
            # Sky handling runs at the processing resolution to match the official
            # threshold semantics.
            depth = _set_sky_regions_to_max_depth(depth=depth, sky=sky)
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            if depth.shape != (orig_h, orig_w):
                depth = F.interpolate(
                    depth[None, None],
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
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


def _set_sky_regions_to_max_depth(*, depth: Tensor, sky: Tensor | None) -> Tensor:
    """Returns depth with sky pixels set to the 99th percentile of non-sky depth.

    Args:
        depth: Depth tensor of shape ``(H, W)``.
        sky: Sky-confidence tensor of shape ``(H, W)``, or None if the model has no
            sky head. Pixels with confidence >= 0.3 are treated as sky.

    Returns:
        The depth tensor with sky pixels replaced, or the input depth unchanged if
        there is no sky output or too few sky or non-sky pixels.
    """
    if sky is None:
        return depth

    non_sky_mask = sky < 0.3
    if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
        return depth

    non_sky_depth = depth[non_sky_mask]
    if non_sky_depth.numel() > 100_000:
        generator = torch.Generator(device=non_sky_depth.device).manual_seed(42)
        idx = torch.randint(
            0,
            non_sky_depth.numel(),
            (100_000,),
            generator=generator,
            device=non_sky_depth.device,
        )
        non_sky_depth = non_sky_depth[idx]
    non_sky_max = torch.quantile(non_sky_depth, 0.99)

    depth = depth.clone()
    depth[~non_sky_mask] = non_sky_max
    return depth


def _load_pretrained_weights(
    model: DepthAnythingV3RelativeDepthEstimation,
    *,
    weights: PathLike | None,
    canonical_name: str,
) -> None:
    """Loads the converted Depth Anything V3 checkpoint into the model in place.

    A local converted ``weights`` path takes precedence. Otherwise the checkpoint is
    resolved by ``canonical_name`` via ``task_model_helpers.download_checkpoint``
    (downloaded to the model cache and verified against its sha256). It is produced by
    ``convert_checkpoint_dav3``.
    """
    checkpoint: PathLike = weights if weights is not None else canonical_name
    checkpoint_path = task_model_helpers.download_checkpoint(checkpoint=checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "train_model" in state:
        state_dict = dict(state["train_model"])
    else:
        state_dict = dict(state)
    model.load_train_state_dict(state_dict)
