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
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

# Modifications Copyright 2026 Lightly AG:
# - Added EdgeCrafter instance segmentation mask output support.
# - Kept encoder auxiliary outputs box/class-only (matching EdgeCrafter, whose
#   ``enc_aux_outputs`` carry no masks).
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lightly_train._task_models.instance_segmentation_components.edgecrafter_head import (
    EdgeCrafterInstanceSegmentationHead,
)
from lightly_train._task_models.object_detection_components.denoising import (
    get_contrastive_denoising_training_group,
)
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)

__all__ = ["EdgeCrafterInstanceSegmentationTransformer"]


class EdgeCrafterInstanceSegmentationTransformer(DFINETransformer):
    """D-FINE transformer that adds per-query instance mask logits."""

    def __init__(
        self,
        *args: Any,
        mask_bottleneck_ratio: int | None = 1,
        mask_downsample_ratio: int = 4,
        mask_spatial_level: int = 0,
        mask_layer_scale_init_value: float = 0.0,
        eval_spatial_size: tuple[int, int],
        **kwargs: Any,
    ) -> None:
        kwargs["eval_spatial_size"] = eval_spatial_size
        super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]

        self.mask_spatial_level = mask_spatial_level
        self.mask_head = EdgeCrafterInstanceSegmentationHead(
            in_dim=self.hidden_dim,
            # One block per query state returned by the decoder. The decoder
            # only emits query states for layers up to ``eval_idx`` (the wider
            # post-``eval_idx`` layers carry mismatched widths), so the block
            # count must match that rather than the full layer count.
            num_blocks=self.decoder.eval_idx + 1,
            bottleneck_ratio=mask_bottleneck_ratio,
            downsample_ratio=mask_downsample_ratio,
            image_size=eval_spatial_size,
            layer_scale_init_value=mask_layer_scale_init_value,
        )

    def forward(
        self,
        feats: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
        spatial_feat: Tensor | None = None,
    ) -> dict[str, Any]:
        proj_feats = self._get_projected_feats(feats)  # type: ignore[no-untyped-call]
        if spatial_feat is None:
            spatial_feat = proj_feats[self.mask_spatial_level]

        memory, spatial_shapes = self._get_encoder_input_from_projected_feats(
            proj_feats
        )  # type: ignore[no-untyped-call]

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(  # type: ignore[no-untyped-call]
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                None,
                None,
                None,
                None,
            )

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
        ) = self._get_decoder_input(  # type: ignore[no-untyped-call]
            memory,
            spatial_shapes,
            denoising_logits,
            denoising_bbox_unact,
        )

        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            query_states,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            return_query_states=True,
        )

        mask_logits = torch.stack(
            self.mask_head(
                spatial_features=spatial_feat,
                query_features=list(query_states.unbind(0)),
            )
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(
                pre_logits, dn_meta["dn_num_split"], dim=1
            )
            dn_pre_bboxes, pre_bboxes = torch.split(
                pre_bboxes, dn_meta["dn_num_split"], dim=1
            )
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta["dn_num_split"], dim=2
            )
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta["dn_num_split"], dim=2
            )
            dn_out_masks, out_masks = torch.split(
                mask_logits, dn_meta["dn_num_split"], dim=2
            )

            dn_out_corners, out_corners = torch.split(
                out_corners, dn_meta["dn_num_split"], dim=2
            )
            dn_out_refs, out_refs = torch.split(
                out_refs, dn_meta["dn_num_split"], dim=2
            )
        else:
            out_masks = mask_logits

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_masks": out_masks[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_masks": out_masks[-1],
            }

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
                outputs_masks=out_masks[:-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(
                enc_topk_logits_list,
                enc_topk_bboxes_list,
            )
            # EdgeCrafter sets ``pre_outputs["pred_masks"] = pre_segs`` where
            # ``pre_segs`` is the final decoder layer's mask (``dec_out_segs[-1]``);
            # after the denoising split that is exactly ``out_masks[-1]``.
            out["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_boxes": pre_bboxes,
                "pred_masks": out_masks[-1],
            }
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                    outputs_masks=dn_out_masks,
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_boxes": dn_pre_bboxes,
                    "pred_masks": dn_out_masks[-1],
                }
                out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class: Tensor,
        outputs_coord: Tensor,
        outputs_corners: Tensor,
        outputs_ref: Tensor,
        teacher_corners: Tensor | None = None,
        teacher_logits: Tensor | None = None,
        outputs_masks: Tensor | None = None,
    ) -> list[dict[str, Tensor | None]]:
        if outputs_masks is None:
            return [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                    "pred_corners": c,
                    "ref_points": d,
                    "teacher_corners": teacher_corners,
                    "teacher_logits": teacher_logits,
                }
                for a, b, c, d in zip(
                    outputs_class,
                    outputs_coord,
                    outputs_corners,
                    outputs_ref,
                )
            ]

        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_masks": m,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d, m in zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref,
                outputs_masks,
            )
        ]
