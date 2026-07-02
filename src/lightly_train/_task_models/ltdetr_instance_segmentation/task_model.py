#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly_train._models import package_helpers
from lightly_train._models.ecvit.ecvit import ECViTModelWrapper
from lightly_train._task_models.dinov3_ltdetr.task_model import (
    _DINOv3LTDETRBase,
)
from lightly_train._task_models.instance_segmentation_components.edgecrafter_decoder import (
    ECSegTransformer,
)
from lightly_train._task_models.instance_segmentation_components.edgecrafter_postprocessor import (
    ECSegPostProcessor,
)
from lightly_train._task_models.ltdetr_instance_segmentation.config import (
    LTDETR_MODEL_REGISTRY,
    SegmentorConfig,
)
from lightly_train._task_models.ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
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
        load_weights: bool = True,
    ) -> None:
        """Create an LTDETR instance segmentation task model.

        Args:
            model_name:
                The model name. For example ``"ltdetrv2-s"`` or
                ``"edgecrafter/ecvitt-ltdetr"``.
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
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        # Bypass _DINOv3LTDETRBase.__init__ (old config system) and call
        # TaskModel.__init__ directly to store init_args for checkpointing.
        super(_DINOv3LTDETRBase, self).__init__(
            init_args=locals(), ignore_args={"load_weights"}
        )

        config: SegmentorConfig = LTDETR_MODEL_REGISTRY.get(alias=model_name)()

        package_name, short_backbone = package_helpers.parse_model_name(
            config.backbone_name
        )
        self.image_size = image_size
        self.classes = classes
        self.backbone_freeze = backbone_freeze

        if backbone_freeze:
            config.backbone_wrapper.finetune = False

        # Use the config's baked-in patch_size unless the caller overrides it.
        if patch_size is None:
            patch_size = config.backbone_args.get("patch_size")
        else:
            config.backbone_args["patch_size"] = patch_size
        config.resolve_auto(patch_size=patch_size)

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
        self.included_classes: dict[int, str] = {
            internal_class_id: class_name
            for internal_class_id, class_name in enumerate(self.classes.values())
        }

        self.image_normalize = image_normalize
        self._expected_input_channels = 3

        # Build backbone model args: start from config defaults, then apply overrides.
        backbone_model_args: dict[str, Any] = dict(config.backbone_args)
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)
        if backbone_weights is not None:
            backbone_model_args["weights"] = str(backbone_weights)

        package = package_helpers.get_package(package_name)

        backbone = package.get_model(
            model_name=short_backbone,
            model_args=backbone_model_args,
            load_weights=load_weights,
        )
        assert isinstance(backbone, ECViTModelWrapper)
        self.backbone: ECViTBackboneWrapper = ECViTBackboneWrapper(
            model_wrapper=backbone
        )

        self.encoder: HybridEncoder = HybridEncoder(
            **config.hybrid_encoder.model_dump()
        )

        transformer_cfg = config.transformer.model_dump()
        transformer_cfg["num_classes"] = len(self.classes)
        self.decoder: ECSegTransformer = ECSegTransformer(  # type: ignore[no-untyped-call]
            **transformer_cfg,
            eval_spatial_size=self.image_size,
        )

        postprocessor_cfg = config.ecseg_postprocessor.model_dump()
        postprocessor_cfg["num_classes"] = len(self.classes)
        self.postprocessor: ECSegPostProcessor = ECSegPostProcessor(**postprocessor_cfg)

        if self.backbone_freeze:
            self.freeze_backbone()

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        return model in LTDETR_MODEL_REGISTRY.list_aliases()

    @classmethod
    def list_model_names(cls) -> list[str]:
        return list(LTDETR_MODEL_REGISTRY.list_aliases())

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
