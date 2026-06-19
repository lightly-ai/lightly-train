#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.dinov2_ltdetr import (
    task_model as _shared_ltdetr_task_model,
)
from lightly_train._task_models.dinov2_ltdetr.task_model import (
    _DINOv2LTDETRBase,
    _DINOv2LTDETRConfig,
    _DINOv2LTDETRViTBConfig,
    _DINOv2LTDETRViTGConfig,
    _DINOv2LTDETRViTLConfig,
    _DINOv2LTDETRViTSConfig,
    _LTDETRDecoderName,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.dinov2_vit_wrapper import (
    DINOv2STAs,
)
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetr_postprocessor import (
    RTDETRPostProcessor,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

# Backwards-compatible private aliases for internal/tests imports. The shared
# config classes moved to ``dinov2_ltdetr.task_model``; keep the old
# object-detection-prefixed names working here.
_DINOv2LTDETRObjectDetectionConfig = _shared_ltdetr_task_model._DINOv2LTDETRConfig
_DINOv2LTDETRObjectDetectionViTSConfig = (
    _shared_ltdetr_task_model._DINOv2LTDETRViTSConfig
)
_DINOv2LTDETRObjectDetectionViTBConfig = (
    _shared_ltdetr_task_model._DINOv2LTDETRViTBConfig
)
_DINOv2LTDETRObjectDetectionViTLConfig = (
    _shared_ltdetr_task_model._DINOv2LTDETRViTLConfig
)
_DINOv2LTDETRObjectDetectionViTGConfig = (
    _shared_ltdetr_task_model._DINOv2LTDETRViTGConfig
)


class DINOv2LTDETRObjectDetection(_DINOv2LTDETRBase):
    def build_decoder(
        self, config: _DINOv2LTDETRConfig
    ) -> RTDETRTransformerv2 | DFINETransformer:
        return _build_decoder(
            config=config,
            decoder_name=config.decoder_name,
            num_classes=len(self.classes),
            image_size=self.image_size,
        )

    def build_postprocessor(self, config: _DINOv2LTDETRConfig) -> RTDETRPostProcessor:
        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        postprocessor_config.update({"num_classes": len(self.classes)})
        return RTDETRPostProcessor(**postprocessor_config)

    def forward_backend(self, x: Tensor) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: Any | dict[str, Tensor],
        metadata: Sequence[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Tensor]]:
        if not isinstance(raw_outputs, dict):
            raise ValueError(
                f"Expected raw_outputs to be a dict, got {type(raw_outputs).__name__}."
            )
        device = next(self.parameters()).device
        # Postprocessor expects (W, H) per image.
        orig_target_size = torch.tensor(
            [[m["orig_w"], m["orig_h"]] for m in metadata],
            dtype=torch.int64,
            device=device,
        )
        postprocessor_out: tuple[Tensor, Tensor, Tensor] = self.postprocessor(
            raw_outputs, orig_target_size
        )
        out: list[dict[str, Tensor]] = []
        labels_batch, boxes_batch, scores_batch = postprocessor_out

        labels_batch = self.internal_class_to_class[labels_batch]
        for i in range(len(metadata)):
            keep = scores_batch[i] > threshold
            out.append(
                {
                    "labels": labels_batch[i][keep],
                    "bboxes": boxes_batch[i][keep],
                    "scores": scores_batch[i][keep],
                }
            )
        return out

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
        labels, boxes, scores = tiling_utils.combine_object_detection_tiles(
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

    def forward(
        self, x: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Function used for ONNX export
        # TODO (Simon, 05/26) This class does not seem to have an export_onnx function
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

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x


class DINOv2LTDETRDSPObjectDetection(DINOv2LTDETRObjectDetection):
    model_suffix = "ltdetr-dsp"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_freeze: bool = False,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        decoder_name: _LTDETRDecoderName = "rtdetrv2",
    ) -> None:
        # Skip the base LTDETR __init__ and register init args directly on
        # TaskModel; this variant builds its own backbone/encoder/decoder below.
        super(_DINOv2LTDETRBase, self).__init__(
            init_args=locals(), ignore_args={"backbone_weights"}
        )
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.image_size = image_size
        self.classes = classes
        self.backbone_freeze = backbone_freeze

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

        if backbone_args is not None and "in_chans" in backbone_args:
            self._expected_input_channels: int = backbone_args["in_chans"]
        elif self.image_normalize is not None:
            self._expected_input_channels = len(self.image_normalize["mean"])
        else:
            self._expected_input_channels = 3

        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
        )

        # Get the configuration based on the model name.
        config_mapping = {
            "vits14": _DINOv2LTDETRViTSConfig,
            "vitb14": _DINOv2LTDETRViTBConfig,
            "vitl14": _DINOv2LTDETRViTLConfig,
            "vitg14": _DINOv2LTDETRViTGConfig,
        }
        config_name = parsed_name["backbone_name"]
        config_cls = config_mapping[config_name]
        config = config_cls()
        config.decoder_name = decoder_name

        model_wrapper = DINOv2ViTModelWrapper(dinov2)
        self.backbone: DINOv2STAs = DINOv2STAs(
            model_wrapper=model_wrapper,
            # Disable STA for DINOv2 as it doesn't work well with patch size 14.
            use_sta=False,
            **config.backbone_wrapper.model_dump(),
        )

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            **config.hybrid_encoder.model_dump()
        )

        self.decoder = _build_decoder(
            config=config,
            decoder_name=config.decoder_name,
            num_classes=len(self.classes),
            image_size=self.image_size,
            cross_attn_method="discrete",
        )

        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_config
        )


def _build_decoder(
    *,
    config: _DINOv2LTDETRConfig,
    decoder_name: _LTDETRDecoderName,
    num_classes: int,
    image_size: tuple[int, int],
    cross_attn_method: str | None = None,
) -> RTDETRTransformerv2 | DFINETransformer:
    if decoder_name == "rtdetrv2":
        decoder_config = config.rtdetr_transformer.model_dump()
        if cross_attn_method is not None:
            decoder_config["cross_attn_method"] = cross_attn_method
        decoder_config.update({"num_classes": num_classes})
        return RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=image_size,
        )
    elif decoder_name == "dfine":
        decoder_config = config.dfine_transformer.model_dump()
        if cross_attn_method is not None:
            decoder_config["cross_attn_method"] = cross_attn_method
        decoder_config.update({"num_classes": num_classes})
        return DFINETransformer(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=image_size,
        )
    else:
        raise ValueError(f"Unsupported LTDETR decoder: {decoder_name}")
