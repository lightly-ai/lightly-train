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

# Depth Anything V3 processes images by resizing the longer side to the inference size.
_PROCESS_RES_METHOD_DAV3 = "upper_bound_resize"
# Depth Anything V3 metric scaling constant: ``metric_depth = focal * output / 300``.
_METRIC_SCALE_FACTOR = 300.0

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2/dav2-relative-small": {
        "canonical_name": "dinov2/dav2-relative-small",
        "backbone_name": "vits14-noreg",
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "relu",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "none",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "relu",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "none",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "relu",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "none",
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
        },
    },
    # The official Depth Anything V2 metric models are trained per domain with a fixed
    # maximum depth that the sigmoid head output is scaled by: 20 m for the indoor
    # Hypersim models and 80 m for the outdoor Virtual KITTI 2 models. The per-size
    # backbone and DPT arguments are identical to the relative models.
    "dinov2/dav2-metric-small-hypersim": {
        "canonical_name": "dinov2/dav2-metric-small-hypersim",
        "backbone_name": "vits14-noreg",
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "sigmoid",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "max_depth",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "sigmoid",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "max_depth",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "sigmoid",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "max_depth",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "sigmoid",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "max_depth",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "sigmoid",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "max_depth",
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
        "image_size": 518,
        "preprocess": "dav2",
        "activation": "sigmoid",
        "use_sky_head": False,
        "align_corners": True,
        "scale_mode": "max_depth",
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
    "dinov2/dav3-relative-large": {
        "canonical_name": "dinov2/dav3-relative-large",
        "backbone_name": "vitl14-noreg",
        "image_size": 504,
        "preprocess": "dav3",
        "activation": "exp",
        "use_sky_head": True,
        "align_corners": False,
        "scale_mode": "none",
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
            "use_sky_head": True,
        },
    },
    "dinov2/dav3-metric-large": {
        "canonical_name": "dinov2/dav3-metric-large",
        "backbone_name": "vitl14-noreg",
        "image_size": 504,
        "preprocess": "dav3",
        "activation": "exp",
        "use_sky_head": True,
        "align_corners": False,
        "scale_mode": "focal",
        "model_args": {
            "out_layers": (4, 11, 17, 23),
            "image_size": 518,
            "patch_size": 14,
            "features": 256,
            "out_channels": (256, 512, 1024, 1024),
            "output_dim": 1,
            "use_sky_head": True,
        },
    },
}


class DepthAnythingDepthEstimation(TaskModel):
    """Depth Anything V2/V3 relative- and metric-depth inference model.

    A single class serving all Depth Anything depth variants. The variant is selected by
    ``model_name`` (see ``list_model_names``); the DINOv2 backbone, DPT head and
    pre/post-processing are configured per variant from the model registry.
    """

    model_suffix = "depth_anything"

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
                The Depth Anything model name. One of the names returned by
                ``list_model_names``, e.g. ``"dinov2/dav2-relative-large"``,
                ``"dinov2/dav2-metric-large-hypersim"``, ``"dinov2/dav3-relative-large"``
                or ``"dinov2/dav3-metric-large"``.
            model_args:
                Additional arguments controlling the DPT decoder and feature extraction,
                e.g. ``out_layers``, ``features``, ``out_channels``, ``output_dim``,
                ``use_sky_head`` (V3) or ``max_depth`` (V2 metric).
            backbone_args:
                Additional arguments passed to the DINOv2 backbone construction (see
                ``DINOV2_VIT_PACKAGE.get_model``), e.g. ``in_chans`` or
                ``drop_path_rate``. These override the Depth Anything defaults.
            load_weights:
                If True, load the converted Depth Anything weights (backbone and DPT
                head) so the model is a ready-to-use depth predictor; a local
                ``weights`` path takes precedence over the hosted URL. If False, the
                model is randomly initialized, e.g. when restoring from an exported
                checkpoint via ``load_train_state_dict``.
            weights:
                Optional path to a converted Depth Anything checkpoint (the
                ``convert_checkpoint_dav2``/``convert_checkpoint_dav3`` output) to load
                instead of the hosted weights. Required for the non-Apache-2.0 V2 models,
                which are not hosted.
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
        # The image size is fixed per model: it is the official Depth Anything
        # inference resolution and predictions are resized back to the original image
        # size, so it is not a user-facing parameter.
        self.image_size = int(config["image_size"])

        self._preprocess: str = config["preprocess"]
        self._align_corners: bool = bool(config["align_corners"])
        self._scale_mode: str = config["scale_mode"]

        net_args = dict(config["model_args"])
        if model_args is not None:
            net_args.update(model_args)

        patch_size = int(net_args["patch_size"])
        self.out_layers: tuple[int, ...] = tuple(net_args["out_layers"])
        self.patch_size = patch_size
        # `max_depth`/`use_sky_head`/`activation` historically live inside `model_args`
        # in exported checkpoints, so read them from the merged `net_args` with the
        # per-config value as the default to stay reconstructable from old checkpoints.
        activation = str(net_args.get("activation", config["activation"]))
        use_sky_head = bool(net_args.get("use_sky_head", config["use_sky_head"]))
        if self._scale_mode == "max_depth":
            # Fixed maximum depth in meters that the sigmoid head output is scaled by.
            self.max_depth = float(net_args["max_depth"])

        # Reproduce the plain (register-free, unchunked, MLP-FFN) ViT that Depth Anything
        # is built on so the state dict keys match the checkpoint; `block_chunks=0` keeps
        # the `blocks.{i}.` key layout. The backbone is built without weights: when
        # `load_weights` is True the converted checkpoint is loaded below instead.
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
            activation=activation,
            use_sky_head=use_sky_head,
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
    def predict(  # type: ignore[override]
        self,
        image: PathLike | PILImage | Tensor,
        *,
        intrinsics: Tensor | None = None,
    ) -> Tensor:
        """Returns a depth map for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape ``(C, H, W)``; uint8 tensors are interpreted in [0, 255] and
                float tensors in [0, 1].
            intrinsics:
                ``(3, 3)`` camera intrinsics matrix of the original image in pixel
                coordinates. Required for the metric V3 model, which returns metric depth
                in meters following the official DA3 formula
                ``metric_depth = focal * output / 300`` (with the focal length rescaled
                to the processing resolution). Must be None for all other models.

        Returns:
            A depth tensor of shape ``(H, W)`` matching the original input resolution.
            For metric models, values are metric depth in meters and larger values
            correspond to farther scene content; for relative models, see the model's
            documentation for the depth convention.
        """
        self._track_inference()
        if self.training:
            self.eval()

        self._validate_intrinsics(intrinsics=intrinsics)
        x, metadata = self.preprocess_image(image)
        if intrinsics is not None:
            metadata["focal"] = _processed_focal_length(
                intrinsics=intrinsics,
                orig_h=metadata["orig_h"],
                orig_w=metadata["orig_w"],
                proc_h=int(x.shape[-2]),
                proc_w=int(x.shape[-1]),
            )
        batch = self.preprocess_batch([x])
        raw = self.forward(batch)
        return self.postprocess(raw, [metadata])[0]

    @torch.no_grad()
    def predict_batch(  # type: ignore[override]
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        *,
        intrinsics: Sequence[Tensor] | None = None,
    ) -> list[Tensor]:
        """Returns depth maps for the given batch of images.

        Args:
            images:
                Sequence of input images. Each can be a path, URL, PIL image, or
                tensor. Tensors must have shape ``(C, H, W)``; uint8 tensors are
                interpreted in [0, 255] and float tensors in [0, 1].
            intrinsics:
                Sequence of ``(3, 3)`` camera intrinsics matrices, one per image, in
                original-image pixel coordinates. Required for the metric V3 model and
                must be None for all other models.

        Returns:
            One depth tensor of shape ``(H, W)`` per image, matching each image's
            original resolution. Images whose processed sizes differ are center-cropped
            to the smallest size in the batch before inference, so their depth maps are
            slightly stretched when resized back.
        """
        self._track_inference()
        if self.training:
            self.eval()

        self._validate_intrinsics(intrinsics=intrinsics)
        if intrinsics is not None and len(intrinsics) != len(images):
            raise ValueError(
                f"Expected one intrinsics matrix per image, got {len(intrinsics)} "
                f"intrinsics for {len(images)} images."
            )

        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for i, image in enumerate(images):
            x, meta = self.preprocess_image(image)
            if intrinsics is not None:
                meta["focal"] = _processed_focal_length(
                    intrinsics=intrinsics[i],
                    orig_h=meta["orig_h"],
                    orig_w=meta["orig_w"],
                    proc_h=int(x.shape[-2]),
                    proc_w=int(x.shape[-1]),
                )
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
        if self._preprocess == "dav3":
            # Process on the input's native device: the cv2-parity resize rounds after an
            # einsum whose accumulation order differs between CPU and GPU, so moving to
            # the model device first could flip pixels and break bit-exactness.
            x = image_utils.process_image_dav3(
                x,
                process_res=self.image_size,
                process_res_method=_PROCESS_RES_METHOD_DAV3,
            )
        else:
            x = image_utils.process_image_dav2(x, process_res=self.image_size)
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
        """Maps raw forward outputs to one depth tensor per image, bilinearly resized to
        the original input size (``orig_h``, ``orig_w`` from the metadata).

        For V3 models the sky pixels are filled at the processing resolution before any
        scaling. For metric models the depth is scaled before the resize: V2 metric
        scales the sigmoid output by the per-model ``max_depth`` (matching the official
        ``head(...) * max_depth`` order); V3 metric scales by ``focal / 300`` when the
        metadata carries a ``focal`` entry.
        """
        depth_batch = raw_outputs["depth"]
        sky_batch = raw_outputs.get("sky")
        out: list[Tensor] = []
        for i, meta in enumerate(metadata):
            depth = depth_batch[i, 0]
            # Sky handling runs at the processing resolution to match the official
            # threshold semantics. It is a no-op when the model has no sky head.
            sky = None if sky_batch is None else sky_batch[i, 0]
            depth = _set_sky_regions_to_max_depth(depth=depth, sky=sky)
            if self._scale_mode == "max_depth":
                depth = depth * self.max_depth
            elif self._scale_mode == "focal":
                focal = meta.get("focal")
                if focal is not None:
                    depth = depth * (focal / _METRIC_SCALE_FACTOR)
            orig_h, orig_w = meta["orig_h"], meta["orig_w"]
            if depth.shape != (orig_h, orig_w):
                depth = F.interpolate(
                    depth[None, None],
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=self._align_corners,
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

    def _validate_intrinsics(
        self, *, intrinsics: Tensor | Sequence[Tensor] | None
    ) -> None:
        if self._scale_mode == "focal":
            if intrinsics is None:
                raise ValueError(
                    "This model requires per-image camera intrinsics; pass "
                    "intrinsics=... (a (3, 3) matrix per image)."
                )
        elif intrinsics is not None:
            raise ValueError("This model does not accept intrinsics.")


def _processed_focal_length(
    *,
    intrinsics: Tensor,
    orig_h: int,
    orig_w: int,
    proc_h: int,
    proc_w: int,
) -> float:
    """Returns the focal length in pixels at the processing resolution.

    Mirrors the official DA3 input processor, which rescales ``fx`` by the width ratio
    and ``fy`` by the height ratio, and the official metric scaling, which uses the
    average of ``fx`` and ``fy``. Center-cropping during batch size unification does
    not change the focal length, so the pre-crop processed size is the right reference.

    Args:
        intrinsics: Camera intrinsics matrix of shape ``(3, 3)`` in original-image
            pixel coordinates.
        orig_h: Original image height.
        orig_w: Original image width.
        proc_h: Processed image height.
        proc_w: Processed image width.

    Returns:
        The focal length in pixels at the processing resolution.
    """
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Expected intrinsics of shape (3, 3), got {tuple(intrinsics.shape)}."
        )
    fx = float(intrinsics[0, 0]) * (proc_w / orig_w)
    fy = float(intrinsics[1, 1]) * (proc_h / orig_h)
    return (fx + fy) / 2


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
    model: DepthAnythingDepthEstimation,
    *,
    weights: PathLike | None,
    canonical_name: str,
) -> None:
    """Loads the converted Depth Anything checkpoint into the model in place.

    A local converted ``weights`` path takes precedence. Otherwise the checkpoint is
    resolved by ``canonical_name`` via ``task_model_helpers.download_checkpoint``, which
    downloads the hosted models (verified against their sha256) and raises with
    local-conversion instructions for the non-commercial V2 models, which are not
    redistributed. The checkpoints are produced by ``convert_checkpoint_dav2`` and
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
