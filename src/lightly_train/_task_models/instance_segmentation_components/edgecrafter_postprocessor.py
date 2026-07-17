#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

# Modifications Copyright 2026 Lightly AG:
# - Added typed interfaces.
# - Renamed the postprocessor to EdgeCrafterInstanceSegmentationPostProcessor.
# - Added a deferred-einsum mask path that gathers the selected query features
#   before the mask einsum, avoiding a GatherElements over the full-resolution
#   mask tensor. The materialized-mask gather is kept as a fallback.

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor

from lightly_train._task_models.object_detection_components.rtdetr_postprocessor import (
    RTDETRPostProcessor,
)

__all__ = ["ECSegPostProcessor"]


class ECSegPostProcessor(RTDETRPostProcessor):
    """Postprocessor for LTDETR instance segmentation outputs."""

    def forward(  # type: ignore[override]
        self,
        outputs: dict[str, Tensor],
        orig_target_sizes: Tensor,
    ) -> (
        list[dict[str, Tensor]]
        | tuple[Tensor, Tensor, Tensor]
        | tuple[Tensor, Tensor, Tensor, Tensor]
    ):
        """Converts raw LTDETR outputs to labels, boxes, masks, and scores.

        Args:
            outputs: Model outputs with ``pred_logits``, ``pred_boxes``, and
                ``pred_masks``.
            orig_target_sizes: Original image sizes as ``(W, H)`` rows.

        Returns:
            In deploy mode, a tuple ``(labels, boxes, scores, masks)``. Otherwise,
            a list of per-image dictionaries with the same fields.
        """
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        mask_pred = outputs.get("pred_masks", None)
        # Deferred-einsum operands (deploy/eval path): the mask head returns the
        # projected spatial and query features instead of the full mask tensor so
        # the query gather happens before the einsum (see forward_deploy).
        mask_spatial = outputs.get("pred_mask_spatial", None)
        mask_query = outputs.get("pred_mask_query", None)
        mask_bias = outputs.get("pred_mask_bias", None)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        masks: Tensor | None = None

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = _mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1,
                index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]),
            )
            if mask_spatial is not None:
                # Gather the selected query embeddings (small (B, Q, Ci) index),
                # then run the einsum — instead of gathering the full-resolution
                # mask tensor with a (B, Q, Hm, Wm) index. The deploy decoder
                # always emits the spatial/query/bias operands together.
                assert mask_query is not None
                selected_query = mask_query.gather(
                    dim=1,
                    index=index.unsqueeze(-1).expand(-1, -1, mask_query.shape[-1]),
                )
                masks = torch.einsum("bchw,bqc->bqhw", mask_spatial, selected_query)
                if mask_bias is not None:
                    masks = masks + mask_bias
            elif mask_pred is not None:
                masks = mask_pred.gather(
                    dim=1,
                    index=index.unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, mask_pred.shape[-2], mask_pred.shape[-1]),
                )
        else:
            if mask_pred is not None or mask_spatial is not None:
                # EdgeCrafter only gathers masks in the focal-loss branch; its
                # softmax branch has no mask path. Fail loudly instead of
                # silently dropping the predicted masks.
                raise NotImplementedError(
                    "Mask postprocessing is only supported with use_focal_loss=True."
                )
            # Return the converted/scaled xyxy boxes, not the raw normalized
            # cxcywh ``boxes`` tensor.
            boxes = bbox_pred
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = bbox_pred.gather(
                    dim=1,
                    index=index.unsqueeze(-1).tile(1, 1, bbox_pred.shape[-1]),
                )

        if self.deploy_mode:
            if masks is not None:
                return labels, boxes, scores, masks
            return labels, boxes, scores

        if self.remap_mscoco_category:
            # Matches the parent RTDETRPostProcessor; the module is provided by
            # the runtime data package only when category remapping is enabled.
            from ...data.dataset import (  # type: ignore[import-not-found]
                mscoco_label2category,
            )

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results: list[dict[str, Tensor]] = []
        if masks is not None:
            for i, (sco, lab, box, mask) in enumerate(
                zip(scores, labels, boxes, masks)
            ):
                result = {"scores": sco, "labels": lab, "boxes": box}
                w, h = orig_target_sizes[i].tolist()
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                result["masks"] = mask > 0.0
                results.append(result)
        else:
            results = [
                {"scores": sco, "labels": lab, "boxes": box}
                for sco, lab, box in zip(scores, labels, boxes)
            ]
        return results


def _mod(a: Tensor, b: int) -> Tensor:
    out: Tensor = a - torch.div(a, b, rounding_mode="floor") * b
    return out
