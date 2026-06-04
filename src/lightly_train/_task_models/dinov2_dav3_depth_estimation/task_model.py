#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._data import file_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.depth_estimation_components.input_processor import (
    InputProcessor,
)
from lightly_train._task_models.dinov2_dav3_depth_estimation.dpt import (
    DPT,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "dinov2/dav3mono-large": {
        "canonical_name": "dinov2/dav3mono-large",
        "backbone_name": "vitl14-noreg",
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

# A single canonical name for now. Aliases mapping to the same config key can be
# added here later to make the model easier to find for users.
_MODEL_ALIASES: dict[str, str] = {
    "dinov2/dav3mono-large": "dinov2/dav3mono-large",
}

_DEFAULT_IMAGE_NORMALIZE = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


class DepthAnythingV3MonocularDepthEstimation(TaskModel):
    """Depth Anything V3 monocular relative-depth inference model."""

    model_suffix = "dav3"

    def __init__(
        self,
        *,
        model_name: str = "dinov2/dav3mono-large",
        image_size: int = 504,
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        model_args: dict[str, Any] | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """
        Args:
            model_name:
                The Depth Anything V3 model name. The only supported name is
                ``"dinov2/dav3mono-large"``.
            image_size:
                Upper bound for the longest image side during inference. The resized
                height and width are rounded to the nearest multiple of the DA3 patch
                size. The official DA3 inference default is 504.
            image_normalize:
                Image normalization parameters. Defaults to ImageNet statistics.
            model_args:
                Additional arguments controlling the DPT decoder and feature
                extraction, e.g. ``out_layers``, ``features``, ``out_channels``,
                ``output_dim``, or ``use_sky_head``.
            backbone_args:
                Additional arguments passed to the DINOv2 backbone construction (see
                ``DINOV2_VIT_PACKAGE.get_model``), e.g. ``in_chans`` or
                ``drop_path_rate``. These override the Depth Anything V3 defaults.
            load_weights:
                If True, the backbone is initialized from the pretrained DINOv2
                ViT-L/14 weights that Depth Anything V3 was fine-tuned from. The DPT
                head is always randomly initialized. To obtain the fully pretrained
                depth model, use ``lightly_train.load_model(<exported .pt>)``; the
                ``convert_checkpoint`` script produces that file from the official
                Depth Anything V3 weights. Set to False to skip the backbone download,
                e.g. when the weights are loaded from an exported checkpoint via
                ``load_train_state_dict``.
        """
        super().__init__(locals(), ignore_args={"load_weights"})
        parsed_name = self.parse_model_name(model_name)
        config = _MODEL_CONFIGS[parsed_name]

        self.model_name = config["canonical_name"]
        self.image_size = image_size
        self.image_normalize = (
            _DEFAULT_IMAGE_NORMALIZE if image_normalize is None else image_normalize
        )

        # Reuse the official Depth Anything V3 input pipeline so that resizing and
        # normalization match the reference implementation exactly. The mono model
        # bounds the longest side and rounds both sides to a patch multiple, then
        # applies the official ImageNet normalization.
        self.process_res_method = "upper_bound_resize"
        self._input_processor = InputProcessor()  # type: ignore[no-untyped-call]

        net_args = dict(config["model_args"])
        if model_args is not None:
            net_args.update(model_args)

        patch_size = int(net_args["patch_size"])
        self.out_layers: tuple[int, ...] = tuple(net_args["out_layers"])
        self.patch_size = patch_size

        # Reuse the DINOv2 backbone owned by the package. The overrides reproduce the
        # plain (register-free, unchunked, MLP-FFN) ViT-L that Depth Anything V3 is
        # built on, so the backbone state dict keys match the official checkpoint.
        # `block_chunks=0` is essential: chunked blocks would change the key layout
        # from `blocks.{i}.` to `blocks.{chunk}.{i}.` and break loading.
        # When `load_weights` is True the backbone is initialized from the DINOv2
        # ViT-L/14 weights that DA3 was fine-tuned from; this is skipped when the model
        # is reconstructed from an exported checkpoint, whose weights are loaded by
        # `load_train_state_dict` instead.
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
            load_weights=load_weights,
        )
        self.decoder = DPT(
            dim_in=int(self.backbone.embed_dim),
            patch_size=patch_size,
            output_dim=int(net_args["output_dim"]),
            features=int(net_args["features"]),
            out_channels=tuple(net_args["out_channels"]),
            use_sky_head=bool(net_args["use_sky_head"]),
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
            A depth tensor of shape ``(H, W)`` at the Depth Anything V3 processing
            resolution (the longest side resized to ``image_size``, both sides rounded
            to a multiple of the patch size). This matches the official ``Prediction``
            resolution. Larger values correspond to farther scene content.
        """
        self._track_inference()
        if self.training:
            self.eval()

        x = self._preprocess_image(image)
        out = self.forward(x.unsqueeze(0))
        return out["depth"][0, 0]

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Run depth inference on a preprocessed batch with shape ``(B, 3, H, W)``."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape (B, C, H, W), got {tuple(x.shape)}."
            )
        feats = self._extract_features(x)
        out = self.decoder(feats=feats, H=x.shape[-2], W=x.shape[-1])
        _set_mono_sky_regions_to_max_depth(out)
        return out

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from an exported LightlyTrain checkpoint.

        The state dict is expected to already match this model's layout (flat
        ``backbone.*`` / ``decoder.*`` keys). Converting the official Depth Anything V3
        weights into this layout is the job of the ``convert_checkpoint`` script.
        """
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
        # Delegate resizing and normalization to the official DA3 `InputProcessor`
        # so the result is bit-faithful to the reference: cv2 INTER_CUBIC (upscale) /
        # INTER_AREA (downscale) resizing in uint8 space and ImageNet normalization.
        proc_input = _as_input_processor_image(image)
        batch, _exts, _intrinsics = self._input_processor(
            [proc_input],
            process_res=self.image_size,
            process_res_method=self.process_res_method,
            num_workers=1,
        )
        first_param = next(self.parameters())
        return batch[0].to(device=first_param.device, dtype=first_param.dtype)


def _set_mono_sky_regions_to_max_depth(out: dict[str, Tensor]) -> None:
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


def _as_input_processor_image(image: PathLike | PILImage | Tensor) -> np.ndarray:
    """Converts an input image to an ``(H, W, C)`` / ``(H, W)`` uint8 array.

    Routes through LightlyTrain's loaders so paths, URLs, PIL images, and tensors are
    all supported, then returns a NumPy image that the official DA3 ``InputProcessor``
    accepts (it applies ``Image.fromarray(...).convert("RGB")``, matching the
    reference RGB conversion).

    Args:
        image: The input image as a path, URL, PIL image, or ``(C, H, W)`` tensor.

    Returns:
        The image as a uint8 array, shaped ``(H, W, C)`` or ``(H, W)`` for a single
        channel.
    """
    tensor = file_helpers.as_image_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(tensor.shape)}.")

    array = tensor.permute(1, 2, 0).cpu().numpy()
    if array.shape[2] == 1:
        # `Image.fromarray` expects a 2D array for single-channel images.
        array = array[:, :, 0]
    return np.ascontiguousarray(array)
