#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly_train._task_models.dinov3_ltdetr.task_model import (
    _DINOv3LTDETRBase,
    _DINOv3LTDETRConfig,
)
from lightly_train._task_models.instance_segmentation_components.edgecrafter_decoder import (
    EdgeCrafterInstanceSegmentationTransformer,
)
from lightly_train._task_models.instance_segmentation_components.edgecrafter_postprocessor import (
    EdgeCrafterInstanceSegmentationPostProcessor,
)
from lightly_train.types import PathLike


class LTDETRInstanceSegmentation(_DINOv3LTDETRBase):
    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        patch_size: int | None = None,
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_freeze: bool = False,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        decoder_name: Literal["dfine"] = "dfine",
        load_weights: bool = True,
    ) -> None:
        """Create a DINOv3 LTDETR task model.

        Args:
            model_name:
                The model name. For example ``"dinov3/vits16-ltdetr"``.
            classes:
                A dict mapping class IDs to class names.
            image_size:
                The input image size.
            patch_size:
                Override for the backbone patch size used in ``resolve_auto``
                (stride computation). If None, the value from the model config's
                ``backbone_args`` is used.
            image_normalize:
                A dict containing normalization statistics with the keys ``"mean"``
                and ``"std"``.
            backbone_freeze:
                Whether to freeze the backbone during training.
            backbone_weights:
                Path to the backbone weights.
            backbone_args:
                Additional arguments merged into the backbone model args (override
                config defaults).
            decoder_name:
                Override the decoder from the model config.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        # Bypass _DINOv3LTDETRBase.__init__ (old config system) and call
        # TaskModel.__init__ directly to store init_args for checkpointing.
        super(_DINOv3LTDETRBase, self).__init__(
            init_args=locals(), ignore_args={"load_weights"}
        )

    def build_decoder(
        self, config: _DINOv3LTDETRConfig
    ) -> EdgeCrafterInstanceSegmentationTransformer:
        decoder_config = config.dfine_transformer.model_dump()
        decoder_config.update({"num_classes": len(self.classes)})
        return EdgeCrafterInstanceSegmentationTransformer(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=self.image_size,
        )

    def build_postprocessor(
        self, config: _DINOv3LTDETRConfig
    ) -> EdgeCrafterInstanceSegmentationPostProcessor:
        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        postprocessor_config.update({"num_classes": len(self.classes)})
        return EdgeCrafterInstanceSegmentationPostProcessor(**postprocessor_config)

    def get_export_output_names(self) -> list[str]:
        return ["labels", "boxes", "masks", "scores"]

    def forward_backend(self, x: Tensor) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)

        # Don't pass ``spatial_feat`` so the decoder uses its projected feature
        # ``proj_feats[0]``. For ViT/ECViT presets the decoder's ``input_proj`` is
        # ``Identity`` (encoder and decoder share ``hidden_dim``), so this matches
        # EdgeCrafter's ``spatial_feat = x[0]``. For ConvNeXt presets the encoder
        # emits more channels than the decoder ``hidden_dim``; using the projected
        # feature gives the mask head the channel count it expects.
        return self.decoder(feats=x)

    def forward(
        self, x: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if orig_target_size is None:
            h, w = x.shape[-2:]
            orig_target_size_ = torch.tensor([[w, h]]).to(x.device)
        else:
            orig_target_size_ = orig_target_size[:, [1, 0]].to(
                device=x.device,
                dtype=torch.int64,
            )

        x = self.forward_backend(x)

        result: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor, Tensor] = (
            self.postprocessor(x, orig_target_size_)
        )
        # Postprocessor must be in deploy mode at this point. It returns only tuples
        # during deploy mode.
        assert isinstance(result, tuple) and len(result) == 4
        labels, boxes, scores, masks = result
        labels = self.internal_class_to_class[labels]
        return (labels, boxes, masks, scores)

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        # See ``forward_backend``: omit ``spatial_feat`` so the decoder uses its
        # projected feature ``proj_feats[0]`` (Identity for ViT/ECViT presets).
        x = self.decoder(feats=x, targets=targets)
        return x

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
        orig_target_size = torch.tensor(
            [[m["orig_w"], m["orig_h"]] for m in metadata],
            dtype=torch.int64,
            device=device,
        )
        postprocessor_out: tuple[Tensor, Tensor, Tensor, Tensor] = self.postprocessor(
            raw_outputs, orig_target_size
        )
        out: list[dict[str, Tensor]] = []
        labels_batch, boxes_batch, scores_batch, masks_batch = postprocessor_out

        labels_batch = self.internal_class_to_class[labels_batch]
        for i, meta in enumerate(metadata):
            keep = scores_batch[i] > threshold
            # The deploy postprocessor returns raw mask logits at the mask-head
            # resolution (image_size // downsample_ratio). Interpolate them back
            # to the original image size and binarize, matching the
            # postprocessor's non-deploy branch and the instance-segmentation
            # `predict` contract (boolean masks of shape (N, orig_h, orig_w)).
            masks = masks_batch[i][keep]
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(int(meta["orig_h"]), int(meta["orig_w"])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            out.append(
                {
                    "labels": labels_batch[i][keep],
                    "bboxes": boxes_batch[i][keep],
                    "masks": masks > 0.0,
                    "scores": scores_batch[i][keep],
                }
            )
        return out
