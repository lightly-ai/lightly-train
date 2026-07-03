#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train import _logging, _torch_testing
from lightly_train._data import file_helpers
from lightly_train._export import onnx_helpers, tensorrt_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
        "backbone_package": "dinov2",
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
    # Trainable Depth Anything V3 relative-depth student on a DINOv2 ViT-S backbone. There
    # is no official small V3 checkpoint, so this variant is built without hosted weights
    # (the DINOv2-pretrained backbone is loaded separately during fine-tuning) and is used
    # to distill the V3 ViT-L teacher. The sky head is sigmoid-activated so the output is a
    # [0, 1] confidence map for BCE distillation. DPT sizing mirrors `dav2-relative-small`.
    "dinov2/dav3-relative-small": {
        "canonical_name": "dinov2/dav3-relative-small",
        "backbone_package": "dinov2",
        "backbone_name": "vits14-noreg",
        "image_size": 504,
        "preprocess": "dav3",
        "activation": "exp",
        "use_sky_head": True,
        "sky_activation": "sigmoid",
        "align_corners": False,
        "scale_mode": "none",
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 518,
            "patch_size": 14,
            "features": 64,
            "out_channels": (48, 96, 192, 384),
            "output_dim": 1,
            "use_sky_head": True,
        },
    },
    # DINOv3 ViT-Tiny/TinyPlus DA3-style relative-depth students. They use the same DPT
    # head topology as the DA3 relative student, adjusted to the DINOv3 16px patch grid
    # and the 192-dim Tiny backbone family. These are trainable configs without hosted
    # full depth checkpoints; training loads the DINOv3-pretrained backbone separately.
    "dinov3/dav3-relative-tiny": {
        "canonical_name": "dinov3/dav3-relative-tiny",
        "backbone_package": "dinov3",
        "backbone_name": "vitt16",
        "image_size": 512,
        "preprocess": "dav3",
        "activation": "exp",
        "use_sky_head": True,
        "sky_activation": "sigmoid",
        "align_corners": False,
        "scale_mode": "none",
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 512,
            "patch_size": 16,
            # The 192-dim Tiny backbone is a small teacher target, so the DPT head is
            # sized down (features 32, halved out_channels) to keep it a modest fraction
            # of the ~5.5 M backbone; the resize/fusion convs scale with out_channels.
            "features": 32,
            "out_channels": (24, 48, 96, 192),
            "output_dim": 1,
            "use_sky_head": True,
        },
    },
    "dinov3/dav3-relative-tiny-plus": {
        "canonical_name": "dinov3/dav3-relative-tiny-plus",
        "backbone_package": "dinov3",
        "backbone_name": "vitt16plus",
        "image_size": 512,
        "preprocess": "dav3",
        "activation": "exp",
        "use_sky_head": True,
        "sky_activation": "sigmoid",
        "align_corners": False,
        "scale_mode": "none",
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            "image_size": 512,
            "patch_size": 16,
            # See `dav3-relative-tiny`: the same slimmed DPT head is used for the
            # TinyPlus student.
            "features": 32,
            "out_channels": (24, 48, 96, 192),
            "output_dim": 1,
            "use_sky_head": True,
        },
    },
    # Test-only V3 config: the real ViT-S backbone with a tiny DPT head and a small
    # processing resolution to keep depth fine-tuning tests fast on CPU. (The 2-block
    # `_vittest14` backbone cannot feed the DPT, which needs four distinct intermediate
    # layers.)
    "dinov2/_vittest14-dav3": {
        "canonical_name": "dinov2/_vittest14-dav3",
        "backbone_package": "dinov2",
        "backbone_name": "vits14-noreg",
        "image_size": 70,
        "preprocess": "dav3",
        "activation": "exp",
        "use_sky_head": True,
        "sky_activation": "sigmoid",
        "align_corners": False,
        "scale_mode": "none",
        "model_args": {
            "out_layers": (2, 5, 8, 11),
            # Backbone img_size matches the DINOv2 checkpoint (pos_embed is interpolated
            # to the actual processing resolution at forward time).
            "image_size": 518,
            "patch_size": 14,
            "features": 16,
            "out_channels": (8, 16, 32, 32),
            "output_dim": 1,
            "use_sky_head": True,
        },
    },
}


class DepthAnythingDepthEstimation(TaskModel):
    """Depth Anything V2/V3 relative- and metric-depth inference model.

    A single class serving all Depth Anything depth variants. The variant is selected by
    ``model_name`` (see ``list_model_names``); the ViT backbone, DPT head and
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
                Additional arguments passed to the backbone package construction, e.g.
                ``in_chans`` or ``drop_path_rate``. These override the Depth Anything
                defaults.
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
        # The official V3 sky head is ReLU-activated (the DPT default). Trainable configs
        # override this to "sigmoid" so the sky output is a [0, 1] confidence map suitable
        # for BCE distillation and the `sky < 0.3` threshold in postprocessing.
        sky_activation = str(
            net_args.get("sky_activation", config.get("sky_activation", "relu"))
        )
        if self._scale_mode == "max_depth":
            # Fixed maximum depth in meters that the sigmoid head output is scaled by.
            self.max_depth = float(net_args["max_depth"])

        # Reproduce the backbone variant expected by each depth config. DINOv2 Depth
        # Anything checkpoints use a plain register-free, unchunked MLP-FFN ViT so the
        # state dict keys match; DINOv3 Tiny students use the package defaults plus the
        # selected patch size. The backbone is built without weights here: when
        # `load_weights` is True the full converted checkpoint is loaded below instead.
        backbone_package = str(config["backbone_package"])
        if backbone_package == "dinov2":
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
            self.backbone_package = DINOV2_VIT_PACKAGE
        elif backbone_package == "dinov3":
            backbone_model_args = {
                "patch_size": patch_size,
            }
            self.backbone_package = DINOV3_PACKAGE
        else:
            raise ValueError(f"Unknown backbone package '{backbone_package}'.")
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)
        # Store the backbone construction so fine-tuning can re-fetch the same backbone
        # with pretrained weights and copy them in.
        self.backbone_name: str = config["backbone_name"]
        self.backbone_model_args: dict[str, Any] = backbone_model_args
        self.backbone = self.backbone_package.get_model(
            model_name=config["backbone_name"],
            model_args=backbone_model_args,
            load_weights=False,
        )
        try:
            mask_token = self.backbone.mask_token  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            # Depth estimation does not use the mask token. We disable grads for it to
            # avoid DDP errors from unused parameters (see image_classification's
            # task_model.py for the same pattern).
            mask_token.requires_grad = False
        self.decoder = DPT(
            dim_in=int(self.backbone.embed_dim),
            patch_size=patch_size,
            output_dim=int(net_args["output_dim"]),
            features=int(net_args["features"]),
            out_channels=tuple(net_args["out_channels"]),
            activation=activation,
            use_sky_head=use_sky_head,
            sky_activation=sky_activation,
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
                patch_size=self.patch_size,
            )
        else:
            x = image_utils.process_image_dav2(
                x, process_res=self.image_size, patch_size=self.patch_size
            )
        device = next(self.parameters()).device
        return x.to(device=device), {"orig_h": image_h, "orig_w": image_w}

    def preprocess_batch(  # type: ignore[override]
        self, batch: Sequence[Tensor]
    ) -> Tensor:
        stacked = image_utils.process_batch(batch)
        return stacked.to(dtype=next(self.parameters()).dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Run depth inference on a preprocessed batch with shape ``(B, 3, H, W)``.

        Returns a fixed-order tuple of ``(depth,)`` for models without a sky head and
        ``(depth, sky)`` for models with one (Depth Anything V3). The tuple output makes
        this method directly ONNX-exportable (ONNX graph outputs must be tensors), so the
        exported graph and this eager forward are the exact same code path.
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        feats = self._extract_features(x)
        out: dict[str, Tensor] = self.decoder(feats=feats, H=x.shape[-2], W=x.shape[-1])
        if self.decoder.use_sky_head:
            return out["depth"], out["sky"]
        return (out["depth"],)

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: tuple[Tensor, ...],
        metadata: Sequence[dict[str, Any]],
    ) -> list[Tensor]:
        """Maps raw forward outputs to one depth tensor per image, bilinearly resized to
        the original input size (``orig_h``, ``orig_w`` from the metadata).

        ``raw_outputs`` is the tuple returned by ``forward``: ``(depth,)`` for models
        without a sky head and ``(depth, sky)`` for models with one.

        For V3 models the sky pixels are filled at the processing resolution before any
        scaling. For metric models the depth is scaled before the resize: V2 metric
        scales the sigmoid output by the per-model ``max_depth`` (matching the official
        ``head(...) * max_depth`` order); V3 metric scales by ``focal / 300`` when the
        metadata carries a ``focal`` entry.
        """
        depth_batch = raw_outputs[0]
        sky_batch = raw_outputs[1] if len(raw_outputs) > 1 else None
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
        """Loads weights from a training-export or converted checkpoint.

        Training exports the full ``DepthEstimationTrain`` module, so the task-model
        weights are nested under a ``model.`` prefix alongside criterion and metric
        buffers; converted checkpoints store the bare task-model keys. When any
        ``model.``-prefixed key is present, the prefix is stripped and all other keys
        (criterion, metrics) are dropped; otherwise the state dict is loaded as-is.
        """
        if any(name.startswith("model.") for name in state_dict):
            state_dict = {
                name[len("model.") :]: param
                for name, param in state_dict.items()
                if name.startswith("model.")
            }
        self.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def export_onnx(
        self,
        out: PathLike,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        height: int | None = None,
        width: int | None = None,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> None:
        """Exports the model to ONNX for inference.

        The export uses a dummy input of shape (batch_size, 3, H, W). The spatial size
        (H, W) is fixed in the ONNX graph: it defaults to the model's processing
        resolution (``self.image_size`` on both sides) but can be overridden via
        ``height``/``width`` (both must be multiples of the patch size, 14). If
        ``dynamic_batch_size`` is True, the ONNX graph has a dynamic batch dimension.

        The graph outputs the raw depth map at processing resolution, plus a sky map for
        models with a sky head (Depth Anything V3). Postprocessing (sky filling, metric
        scaling, and resizing back to the original resolution) is not part of the graph
        and must be applied by the caller.

        Optionally simplifies the exported model in-place using onnxslim and verifies
        numerical closeness against a float32 CPU reference via ONNX Runtime.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. Either "auto", "fp32", or "fp16". "auto"
                uses the model's current precision.
            batch_size:
                Batch size for the ONNX input.
            dynamic_batch_size:
                If True, the ONNX graph will have a dynamic batch dimension for the
                input. If False, the batch dimension is fixed to `batch_size`.
            height:
                Height of the ONNX input. If None, will be taken from `self.image_size`.
            width:
                Width of the ONNX input. If None, will be taken from `self.image_size`.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to `torch.onnx.export`.

        Returns:
            None. Writes the ONNX model to `out`.
        """
        from lightly_train._commands import _warnings

        _logging.set_up_console_logging()
        _warnings.filter_export_warnings()

        self.eval()

        first_parameter = next(self.parameters())
        model_device = first_parameter.device
        dtype = first_parameter.dtype

        if precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision != "auto":
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'auto', 'fp32', 'fp16'."
            )

        self.to(dtype)

        height = self.image_size if height is None else height
        width = self.image_size if width is None else width
        num_channels = 3

        if dynamic_batch_size:
            batch_size = 2
        dynamic_axes = {"images": {0: "N"}} if dynamic_batch_size else None

        dummy_input = torch.randn(
            batch_size,
            num_channels,
            height,
            width,
            requires_grad=False,
            device=model_device,
            dtype=dtype,
        )

        # `forward` returns a fixed-order tuple of (depth[, sky]) matching these output
        # names, so it is directly ONNX-exportable without a wrapper.
        output_names = ["depth", "sky"] if self.decoder.use_sky_head else ["depth"]

        # Precalculate interpolated positional encoding for ONNX export.
        with onnx_helpers.precalculate_for_onnx_export():
            self(dummy_input)

        input_names = ["images"]

        torch.onnx.export(
            self,
            (dummy_input,),
            str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes=dynamic_axes,
            **(format_args or {}),
        )

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            # Simplify.
            onnxslim.slim(
                model=str(out),
                output_model=out,
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(out, full_check=True)

            # Always run the reference input in float32 and on cpu for consistency.
            reference_model = copy.deepcopy(self).cpu().to(torch.float32).eval()
            reference_outputs = reference_model(
                dummy_input.cpu().to(torch.float32),
            )

            # Get outputs from the ONNX model.
            session = ort.InferenceSession(out)
            input_feed = {
                "images": dummy_input.cpu().numpy(),
            }
            outputs_onnx = session.run(output_names=None, input_feed=input_feed)
            outputs_onnx = tuple(torch.from_numpy(y) for y in outputs_onnx)

            # Verify that the outputs from both models are close.
            if len(outputs_onnx) != len(reference_outputs):
                raise AssertionError(
                    f"Number of onnx outputs should be {len(reference_outputs)} but is {len(outputs_onnx)}"
                )
            for output_onnx, output_model, output_name in zip(
                outputs_onnx, reference_outputs, output_names
            ):

                def msg(s: str) -> str:
                    return f'ONNX validation failed for output "{output_name}": {s}'

                if output_model.is_floating_point():
                    # Absolute and relative tolerances are a bit arbitrary and taken from here:
                    # https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_core.py#L1611-L1618
                    torch.testing.assert_close(
                        output_onnx,
                        output_model,
                        msg=msg,
                        equal_nan=True,
                        check_device=False,
                        check_dtype=False,
                        check_layout=False,
                        atol=5e-3,
                        rtol=1e-1,
                    )
                else:
                    _torch_testing.assert_most_equal(
                        output_onnx,
                        output_model,
                        msg=msg,
                    )

        logger.info(f"Successfully exported ONNX model to '{out}'")

    @torch.no_grad()
    def export_tensorrt(
        self,
        out: PathLike,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        onnx_args: dict[str, Any] | None = None,
        max_batchsize: int = 1,
        opt_batchsize: int = 1,
        min_batchsize: int = 1,
        verbose: bool = False,
    ) -> None:
        """Build a TensorRT engine from an ONNX model.

        .. note::
            TensorRT is not part of LightlyTrain’s dependencies and must be installed separately.
            Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup.
            See the `TensorRT documentation <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>`_ for more details.
            On CUDA 12.x systems you can often install the Python package via `pip install tensorrt-cu12`.

        This loads the ONNX file, parses it with TensorRT, infers the static input
        shape (C, H, W) from the `"images"` input, and creates an engine with a
        dynamic batch dimension in the range `[min_batchsize, opt_batchsize, max_batchsize]`.
        Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

        The engine is serialized and written to `out`.

        Args:
            out:
                Path where the TensorRT engine will be saved.
            precision:
                Precision for ONNX export and TensorRT engine building. Either
                "auto", "fp32", or "fp16". "auto" uses the model's current precision.
            onnx_args:
                Optional arguments to pass to `export_onnx` when exporting
                the ONNX model prior to building the TensorRT engine. If None,
                default arguments are used and the ONNX file is saved alongside
                the TensorRT engine with the same name but `.onnx` extension.
            max_batchsize:
                Maximum supported batch size.
            opt_batchsize:
                Batch size TensorRT optimizes for.
            min_batchsize:
                Minimum supported batch size.
            verbose:
                Enable verbose TensorRT logging.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
            RuntimeError: If the ONNX cannot be parsed or engine building fails.
            ValueError: If batch size constraints are invalid or H/W are dynamic.
        """
        model_dtype = next(self.parameters()).dtype

        onnx_args = dict(onnx_args) if onnx_args is not None else {}
        onnx_args.setdefault("precision", precision)

        tensorrt_helpers.export_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=model_dtype,
            onnx_args=onnx_args,
            max_batchsize=max_batchsize,
            opt_batchsize=opt_batchsize,
            min_batchsize=min_batchsize,
            # FP32 attention scores required for FP16 model stability. Otherwise output
            # contains NaN.
            fp32_attention_scores=True,
            verbose=verbose,
        )

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


def get_model_image_size(model_name: str) -> int:
    """Returns the fixed processing image size for a depth model.

    Used to resolve the training transform image size from the model name without
    instantiating the model.
    """
    key = model_name.lower()
    if key not in _MODEL_CONFIGS:
        raise ValueError(
            f"Model name '{model_name}' is not supported. Available models are: "
            f"{DepthAnythingDepthEstimation.list_model_names()}."
        )
    return int(_MODEL_CONFIGS[key]["image_size"])


def get_model_patch_size(model_name: str) -> int:
    """Returns the ViT patch size for a depth model."""
    key = model_name.lower()
    if key not in _MODEL_CONFIGS:
        raise ValueError(
            f"Model name '{model_name}' is not supported. Available models are: "
            f"{DepthAnythingDepthEstimation.list_model_names()}."
        )
    return int(_MODEL_CONFIGS[key]["model_args"]["patch_size"])


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
