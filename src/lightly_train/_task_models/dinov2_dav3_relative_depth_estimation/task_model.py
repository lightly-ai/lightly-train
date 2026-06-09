#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._data import cache, download, file_helpers
from lightly_train._env import Env
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.depth_estimation_components.input_processor import (
    InputProcessor,
)
from lightly_train._task_models.dinov2_dav3_relative_depth_estimation.dpt import (
    DPT,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2/dav3-relative-large": {
        "canonical_name": "dinov2/dav3-relative-large",
        "backbone_name": "vitl14-noreg",
        # TODO(Nauryzbay, 06/2026): Host the converted checkpoint and set its URL so
        # `load_weights=True` can download it. Until then pass a local `weights` path.
        "weights_url": None,
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

    model_suffix = "dav3_relative_large"

    def __init__(
        self,
        *,
        model_name: str = "dinov2/dav3-relative-large",
        process_resolution: int = 504,
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
            process_resolution:
                Upper bound for the longest image side during inference. The resized
                height and width are rounded to the nearest multiple of the DA3 patch
                size. The official DA3 inference default is 504.
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
                ``convert_checkpoint`` output) to load instead of the hosted weights.
                Intended for debugging before the checkpoint is hosted.
        """
        super().__init__(locals(), ignore_args={"load_weights", "weights"})
        parsed_name = self.parse_model_name(model_name)
        config = _MODEL_CONFIGS[parsed_name]

        self.model_name = config["canonical_name"]
        self.process_resolution = process_resolution

        self.process_res_method = "upper_bound_resize"
        self._input_processor = InputProcessor()  # type: ignore[no-untyped-call]

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
                weights_url=config["weights_url"],
                weights=weights,
            )

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
        if key in _MODEL_CONFIGS:
            return key
        raise ValueError(
            f"Model name '{model_name}' is not supported. Available models are: "
            f"{cls.list_model_names()}."
        )

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> Tensor:
        """Returns a relative-depth map for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape ``(C, H, W)``.

        Returns:
            A depth tensor of shape ``(H, W)`` at the Depth Anything V3 processing
            resolution (the longest side resized to ``process_resolution``, both sides rounded
            to a multiple of the patch size). This matches the official ``Prediction``
            resolution. Larger values correspond to farther scene content.
        """
        self._track_inference()
        if self.training:
            self.eval()

        x = self._preprocess_image(image)
        out = self.forward(x.unsqueeze(0))
        _set_sky_regions_to_max_depth(out)
        # TODO(Nauryzbay, 06/2026): Resize the depth map back to the original input
        # (H, W) before returning. The official DA3 inference keeps the depth at the
        # processing resolution and we mirror that here to stay close to their code,
        # but the public `predict` API and users expect the original input resolution.
        return out["depth"][0, 0]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run depth inference on a preprocessed batch with shape ``(B, 3, H, W)``."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        feats = self._extract_features(x)
        out: dict[str, Tensor] = self.decoder(feats=feats, H=x.shape[-2], W=x.shape[-1])
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

    def _preprocess_image(self, image: PathLike | PILImage | Tensor) -> Tensor:
        proc_input = _as_input_processor_image(image)
        batch, _exts, _intrinsics = self._input_processor(
            [proc_input],
            process_res=self.process_resolution,
            process_res_method=self.process_res_method,
        )
        first_param = next(self.parameters())
        return batch[0].to(device=first_param.device, dtype=first_param.dtype)


def _set_sky_regions_to_max_depth(out: dict[str, Tensor]) -> None:
    if "depth" not in out or "sky" not in out:
        return

    non_sky_mask = out["sky"] < 0.3
    if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
        return

    non_sky_depth = out["depth"][non_sky_mask]
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

    depth = out["depth"].clone()
    depth[~non_sky_mask] = non_sky_max
    out["depth"] = depth


def _load_pretrained_weights(
    model: DepthAnythingV3RelativeDepthEstimation,
    *,
    weights_url: str | None,
    weights: PathLike | None,
) -> None:
    """Loads the converted Depth Anything V3 checkpoint into the model in place.

    A local ``weights`` path takes precedence; otherwise the checkpoint is downloaded
    from ``weights_url`` into the model cache. Both come from ``convert_checkpoint``.
    """
    if weights is not None:
        checkpoint_path = Path(weights).expanduser()
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint file '{checkpoint_path}' does not exist.")
    elif weights_url is not None:
        checkpoint_path = cache.get_model_cache_dir() / Path(weights_url).name
        if not checkpoint_path.exists():
            logger.info(
                f"Downloading Depth Anything V3 weights from '{weights_url}' to "
                f"'{checkpoint_path}'."
            )
            download.download_from_url(
                weights_url,
                checkpoint_path,
                timeout=Env.LIGHTLY_TRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC.value,
            )
        else:
            logger.info(
                f"Using cached Depth Anything V3 weights from '{checkpoint_path}'."
            )
    else:
        raise RuntimeError(
            "No pretrained Depth Anything V3 checkpoint is available yet: the hosted "
            "weights URL is not set. Pass `weights=<converted .pt>` (produced by the "
            "convert_checkpoint script) to load a local checkpoint, or set "
            "`load_weights=False`."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping) and "train_model" in checkpoint:
        state_dict = dict(checkpoint["train_model"])
    else:
        state_dict = dict(checkpoint)
    model.load_train_state_dict(state_dict)


def _as_input_processor_image(
    image: PathLike | PILImage | Tensor,
) -> NDArray[np.uint8]:
    """Converts an input image to an ``(H, W, C)`` / ``(H, W)`` uint8 array.

    Routes through LightlyTrain's loaders (paths, URLs, PIL images, tensors) and
    returns a NumPy image that the official DA3 ``InputProcessor`` accepts.
    """
    tensor = file_helpers.as_image_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(tensor.shape)}.")

    array = tensor.permute(1, 2, 0).cpu().numpy()
    if array.shape[2] == 1:
        # `Image.fromarray` expects a 2D array for single-channel images.
        array = array[:, :, 0]
    return np.ascontiguousarray(array)
