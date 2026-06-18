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

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor

from lightly_train._task_models.object_detection_components.rtdetr_postprocessor import (
    RTDETRPostProcessor,
)

__all__ = ["EdgeCrafterInstanceSegmentationPostProcessor"]


class EdgeCrafterInstanceSegmentationPostProcessor(RTDETRPostProcessor):
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
            if mask_pred is not None:
                masks = mask_pred.gather(
                    dim=1,
                    index=index.unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, mask_pred.shape[-2], mask_pred.shape[-1]),
                )
        else:
            scores = F.softmax(logits, dim=-1)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes,
                    dim=1,
                    index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]),
                )

        if self.deploy_mode:
            if masks is not None:
                return labels, boxes, scores, masks
            return labels, boxes, scores

        if self.remap_mscoco_category:
            data_dataset = __import__(
                "lightly_train.data.dataset",
                fromlist=["mscoco_label2category"],
            )
            mscoco_label2category = data_dataset.mscoco_label2category
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
                )
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
