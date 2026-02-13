#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
from typing import Any

import torch
from PIL.Image import Image as PILImage
from pydantic import Field
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.dinov2_ltdetr_object_detection.dinov2_vit_wrapper import (
    DINOv2ViTWrapper,
)
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetr_postprocessor import (
    RTDETRPostProcessor,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class _BackboneWrapperConfig(PydanticConfig):
    keep_indices: list[int]


class _BackboneWrapperViTSConfig(_BackboneWrapperConfig):
    keep_indices: list[int] = [5, 8, 11]


class _BackboneWrapperViTBConfig(_BackboneWrapperConfig):
    keep_indices: list[int] = [5, 8, 11]


class _BackboneWrapperViTLConfig(_BackboneWrapperConfig):
    keep_indices: list[int] = [11, 17, 23]


class _BackboneWrapperViTGConfig(_BackboneWrapperConfig):
    keep_indices: list[int] = [19, 29, 39]


class _HybridEncoderConfig(PydanticConfig):
    in_channels: list[int]
    feat_strides: list[int]
    hidden_dim: int
    use_encoder_idx: list[int]
    num_encoder_layers: int
    nhead: int
    dim_feedforward: int
    dropout: float
    enc_act: str
    expansion: float
    depth_mult: float
    act: str
    upsample: bool


class _HybridEncoderViTSConfig(_HybridEncoderConfig):
    in_channels: list[int] = [384, 384, 384]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 384
    use_encoder_idx: list[int] = [2]
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 1536
    dropout: float = 0.0
    enc_act: str = "gelu"
    expansion: float = 1.0
    depth_mult: float = 1
    act: str = "silu"
    upsample: bool = False


class _HybridEncoderViTBConfig(_HybridEncoderConfig):
    in_channels: list[int] = [768, 768, 768]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 768
    use_encoder_idx: list[int] = [2]
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 3072
    dropout: float = 0.0
    enc_act: str = "gelu"
    expansion: float = 1.0
    depth_mult: float = 1
    act: str = "silu"
    upsample: bool = False


class _HybridEncoderViTLConfig(_HybridEncoderConfig):
    in_channels: list[int] = [1024, 1024, 1024]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 1024
    use_encoder_idx: list[int] = [2]
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 4096
    dropout: float = 0.0
    enc_act: str = "gelu"
    expansion: float = 1.0
    depth_mult: float = 1
    act: str = "silu"
    upsample: bool = False


class _HybridEncoderViTGConfig(_HybridEncoderConfig):
    in_channels: list[int] = [1536, 1536, 1536]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 1536
    use_encoder_idx: list[int] = [2]
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 6144
    dropout: float = 0.0
    enc_act: str = "gelu"
    expansion: float = 1.0
    depth_mult: float = 1
    act: str = "silu"
    upsample: bool = False


class _RTDETRTransformerv2Config(PydanticConfig):
    feat_channels: list[int]
    feat_strides: list[int]
    hidden_dim: int
    num_levels: int
    num_layers: int
    num_queries: int
    num_denoising: int
    label_noise_ratio: float
    box_noise_scale: float
    eval_idx: int
    num_points: list[int]
    query_select_method: str


class _RTDETRTransformerv2ViTSConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 256
    num_levels: int = 3
    num_layers: int = 6
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [4, 4, 4]
    query_select_method: str = "default"


class _RTDETRTransformerv2ViTBConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [768, 768, 768]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 256
    num_levels: int = 3
    num_layers: int = 6
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [4, 4, 4]
    query_select_method: str = "default"


class _RTDETRTransformerv2ViTLConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [1024, 1024, 1024]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 256
    num_levels: int = 3
    num_layers: int = 6
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [4, 4, 4]
    query_select_method: str = "default"


class _RTDETRTransformerv2ViTGConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [1536, 1536, 1536]
    feat_strides: list[int] = [14, 14, 14]
    hidden_dim: int = 256
    num_levels: int = 3
    num_layers: int = 6
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [4, 4, 4]
    query_select_method: str = "default"


class _RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class _DINOv2LTDETRObjectDetectionConfig(PydanticConfig):
    backbone_wrapper: _BackboneWrapperConfig
    hybrid_encoder: _HybridEncoderConfig
    rtdetr_transformer: _RTDETRTransformerv2Config
    rtdetr_postprocessor: _RTDETRPostProcessorConfig


class _DINOv2LTDETRObjectDetectionViTSConfig(_DINOv2LTDETRObjectDetectionConfig):
    backbone_wrapper: _BackboneWrapperViTSConfig = Field(
        default_factory=_BackboneWrapperViTSConfig
    )
    hybrid_encoder: _HybridEncoderViTSConfig = Field(
        default_factory=_HybridEncoderViTSConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTSConfig = Field(
        default_factory=_RTDETRTransformerv2ViTSConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv2LTDETRObjectDetectionViTBConfig(_DINOv2LTDETRObjectDetectionConfig):
    backbone_wrapper: _BackboneWrapperViTBConfig = Field(
        default_factory=_BackboneWrapperViTBConfig
    )
    hybrid_encoder: _HybridEncoderViTBConfig = Field(
        default_factory=_HybridEncoderViTBConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTBConfig = Field(
        default_factory=_RTDETRTransformerv2ViTBConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv2LTDETRObjectDetectionViTLConfig(_DINOv2LTDETRObjectDetectionConfig):
    backbone_wrapper: _BackboneWrapperViTLConfig = Field(
        default_factory=_BackboneWrapperViTLConfig
    )
    hybrid_encoder: _HybridEncoderViTLConfig = Field(
        default_factory=_HybridEncoderViTLConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTLConfig = Field(
        default_factory=_RTDETRTransformerv2ViTLConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv2LTDETRObjectDetectionViTGConfig(_DINOv2LTDETRObjectDetectionConfig):
    backbone_wrapper: _BackboneWrapperViTGConfig = Field(
        default_factory=_BackboneWrapperViTGConfig
    )
    hybrid_encoder: _HybridEncoderViTGConfig = Field(
        default_factory=_HybridEncoderViTGConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTGConfig = Field(
        default_factory=_RTDETRTransformerv2ViTGConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class DINOv2LTDETRObjectDetection(TaskModel):
    model_suffix = "ltdetr"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, Any] | None = None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"backbone_weights", "load_weights"}
        )
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.image_size = image_size
        self.classes = classes

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

        self.image_normalize = image_normalize

        # Instantiate the backbone.
        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
            load_weights=load_weights,
        )

        # Optionally load the backbone weights.
        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(dinov2, backbone_weights)

        # Get the configuration based on the model name.
        config_mapping = {
            "vits14": _DINOv2LTDETRObjectDetectionViTSConfig,
            "vitb14": _DINOv2LTDETRObjectDetectionViTBConfig,
            "vitl14": _DINOv2LTDETRObjectDetectionViTLConfig,
            "vitg14": _DINOv2LTDETRObjectDetectionViTGConfig,
        }
        config_name = parsed_name["backbone_name"]
        for key in config_mapping:
            if config_name.startswith(key):
                config_cls = config_mapping[key]
                break
        config = config_cls()

        self.backbone: DINOv2ViTWrapper = DINOv2ViTWrapper(
            model=dinov2,
            **config.backbone_wrapper.model_dump(),
        )
        # TODO(Lionel, 07/25): Improve how mask tokens are handled for fine-tuning.
        # Should we drop them from the model? We disable grads here for DDP to work
        # without find_unused_parameters=True.
        self.backbone.backbone.mask_token.requires_grad = False

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            **config.hybrid_encoder.model_dump()
        )

        decoder_config = config.rtdetr_transformer.model_dump()
        decoder_config.update({"num_classes": len(self.classes)})
        self.decoder: RTDETRTransformerv2 = RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=self.image_size,  # From global config, otherwise anchors are not generated.
        )

        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        postprocessor_config.update({"num_classes": len(self.classes)})
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_config
        )

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            cls.parse_model_name(model_name=model)
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def parse_model_name(cls, model_name: str) -> dict[str, str]:
        def raise_invalid_name() -> None:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available "
                f"models are: {cls.list_model_names()}."
            )

        if not model_name.endswith(f"-{cls.model_suffix}"):
            raise_invalid_name()

        backbone_name = model_name[: -len(f"-{cls.model_suffix}")]

        try:
            package_name, backbone_name = package_helpers.parse_model_name(
                backbone_name
            )
        except ValueError:
            raise_invalid_name()

        if package_name != DINOV2_VIT_PACKAGE.name:
            raise_invalid_name()

        try:
            backbone_name = DINOV2_VIT_PACKAGE.parse_model_name(
                model_name=backbone_name
            )
        except ValueError:
            raise_invalid_name()

        return {
            "model_name": f"{DINOV2_VIT_PACKAGE.name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
        }

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}"
            for name in DINOV2_VIT_PACKAGE.list_model_names()
        ]

    def load_backbone_weights(self, backbone: Module, path: PathLike) -> None:
        """
        Load backbone weights from a checkpoint file.

        Args:
            backbone: backbone to load the statedict in.
            path: path to a .pt file, e.g., exported_last.pt.
        """
        # Check if the file exists.
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}")
            return

        # Load the checkpoint.
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Load the state dict into the backbone.
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)

        # Log missing and unexpected keys.
        if missing or unexpected:
            if missing:
                logger.warning(f"Missing keys when loading backbone: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading backbone: {unexpected}")
        else:
            logger.info("Backbone weights loaded successfully.")

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load the state dict from a training checkpoint.

        Loads the EMA weights if available, otherwise falls back to the model weights.
        """
        has_ema_weights = any(k.startswith("ema_model.model.") for k in state_dict)
        has_model_weights = any(k.startswith("model.") for k in state_dict)
        new_state_dict = {}
        if has_ema_weights:
            for name, param in state_dict.items():
                if name.startswith("ema_model.model."):
                    name = name[len("ema_model.model.") :]
                    new_state_dict[name] = param
        elif has_model_weights:
            for name, param in state_dict.items():
                if name.startswith("model."):
                    name = name[len("model.") :]
                    new_state_dict[name] = param
        return self.load_state_dict(new_state_dict, strict=strict, assign=assign)

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.6
    ) -> dict[str, Tensor]:
        """Run inference on a single image and return detected boxes, labels and scores.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions with
                scores <= threshold are discarded.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in
                  (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the
                  original image.
                - "scores": Tensor of shape (N,) with confidence scores for each prediction.
        """
        self._track_inference()
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()

        first_param = next(self.parameters())
        device = first_param.device
        dtype = first_param.dtype

        # Load image
        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        # Normalize the image.
        if self.image_normalize is not None:
            x = transforms_functional.normalize(
                x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
            )
        x = transforms_functional.resize(x, self.image_size)
        x = x.unsqueeze(0)

        # Select high-confidence predictions. Noteworthy that the selection approach
        # flattens the first two dimensions and would not work with batchsize > 1.
        labels, boxes, scores = self(
            x, orig_target_size=torch.tensor([[image_h, image_w]], device=device)
        )
        keep = scores > threshold
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    @torch.no_grad()
    def predict_sahi(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.6,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
    ) -> dict[str, Tensor]:
        """Run Slicing Aided Hyper Inference (SAHI) inference on the input image.

        The image is first converted to a tensor, then:

        - Tiled into overlapping crops of size `self.image_size`.
        - A resized full-image version is added as a "global" tile.
        - All tiles (global + local) are passed through the model in parallel.
        - Predictions are filtered by score and merged using NMS and a global/local
          consistency heuristic. NMS is only applied on tiles predictions.
          The heuristic discards tiles predictions that heavily overlaps with global
          predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold for filtering low-confidence predictions.
            overlap:
                Fractional overlap between tiles in [0, 1). 0.0 means no overlap.
            nms_iou_threshold:
                IoU threshold used for non-maximum suppression when merging predictions
                from tiles and global image. A lower nms_iou_threshold value yields less
                predictions.
            global_local_iou_threshold:
                Minimum IoU required to consider a tile prediction as matching a global
                prediction when combining them. A lower global_local_iou_threshold
                yields less predictions.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in
                    (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the original image.
                - "scores": Tensor of shape (N,) with confidence scores for each prediction.
        """

        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)

        # Tile the image.
        tiles, tiles_coordinates = tiling_utils.tile_image(x, overlap, self.image_size)

        # Prepare the full image tile
        h, w = x.shape[-2:]
        x = transforms_functional.resize(x, self.image_size)
        x = x.unsqueeze(0)
        tiles = torch.cat([x, tiles], dim=0)

        # Normalize the tiles and the image together.
        tiles = transforms_functional.to_dtype(tiles, dtype=torch.float32, scale=True)

        # Normalize the image.
        if self.image_normalize is not None:
            tiles = transforms_functional.normalize(
                tiles,
                mean=self.image_normalize["mean"],
                std=self.image_normalize["std"],
            )

        # Prepare the image/tiles sizes.
        orig_target_sizes = torch.tensor([self.image_size], device=device).repeat(
            len(tiles), 1
        )
        orig_target_sizes[0, 0] = h
        orig_target_sizes[0, 1] = w

        # Feed the tiles in parallel to the model.
        labels, boxes, scores = self(tiles, orig_target_size=orig_target_sizes)

        # Add coordinates of the tiles to the boxes.
        tiles_coordinates = (
            tiles_coordinates.repeat(1, 2).unsqueeze(1).expand(-1, boxes.shape[1], -1)
        )
        boxes[1:] += tiles_coordinates

        # Reorganize the predictions.
        boxes_global = boxes[0].view(-1, 4)
        boxes_tiles = boxes[1:].view(-1, 4)
        labels_global = labels[0].flatten()
        labels_tiles = labels[1:].flatten()
        scores_global = scores[0].flatten()
        scores_tiles = scores[1:].flatten()

        # Discard low-confidence predictions.
        keep_global = scores_global > threshold
        keep_tiles = scores_tiles > threshold

        # Combine global and tiles predictions.
        labels, boxes, scores = tiling_utils.combine_predictions_tiles_and_global(
            pred_global={
                "labels": labels_global[keep_global],
                "bboxes": boxes_global[keep_global],
                "scores": scores_global[keep_global],
            },
            pred_tiles={
                "labels": labels_tiles[keep_tiles],
                "bboxes": boxes_tiles[keep_tiles],
                "scores": scores_tiles[keep_tiles],
            },
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )

        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    def deploy(self) -> Self:
        self.eval()
        self.postprocessor.deploy()  # type: ignore[no-untyped-call]
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        return self

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x

    def forward(
        self, x: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Function used for ONNX export
        if orig_target_size is None:
            h, w = x.shape[-2:]
            orig_target_size_ = torch.tensor([[w, h]]).to(x.device)
        else:
            # Flip from (H, W) to (W, H).
            orig_target_size = orig_target_size[:, [1, 0]]

            # Move to device.
            orig_target_size_ = orig_target_size.to(device=x.device, dtype=torch.int64)

        # Forward the image through the model.
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)

        result: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor] = (
            self.postprocessor(x, orig_target_size_)
        )
        # Postprocessor must be in deploy mode at this point. It returns only tuples
        # during deploy mode.
        assert isinstance(result, tuple)
        labels, boxes, scores = result
        labels = self.internal_class_to_class[labels]
        return (labels, boxes, scores)


class DINOv2LTDETRDSPObjectDetection(DINOv2LTDETRObjectDetection):
    model_suffix = "ltdetr-dsp"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, Any] | None = None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        super(DINOv2LTDETRObjectDetection, self).__init__(
            init_args=locals(), ignore_args={"backbone_weights"}
        )
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.image_size = image_size
        self.classes = classes

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

        self.image_normalize = image_normalize

        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
        )

        # Get the configuration based on the model name.
        config_mapping = {
            "vits14": _DINOv2LTDETRObjectDetectionViTSConfig,
            "vitb14": _DINOv2LTDETRObjectDetectionViTBConfig,
            "vitl14": _DINOv2LTDETRObjectDetectionViTLConfig,
            "vitg14": _DINOv2LTDETRObjectDetectionViTGConfig,
        }
        config_name = parsed_name["backbone_name"]
        config_cls = config_mapping[config_name]
        config = config_cls()

        self.backbone: DINOv2ViTWrapper = DINOv2ViTWrapper(
            model=dinov2,
            **config.backbone_wrapper.model_dump(),
        )

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            **config.hybrid_encoder.model_dump()
        )

        decoder_config = config.rtdetr_transformer.model_dump()
        decoder_config.update({"cross_attn_method": "discrete"})
        self.decoder: RTDETRTransformerv2 = RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=(
                644,
                644,
            ),  # From global config, otherwise anchors are not generated.
        )

        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_config
        )
