#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import random

from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
    DINOv2ViTTransformArgs,
)
from lightly_train._methods.dinov31.constrained_crop import (
    render_augmented_local,
    render_clean_global,
    sample_high_overlap_box,
)
from lightly_train.types import TransformInput, TransformOutput


class DINOv31TransformArgs(DINOv2ViTTransformArgs):
    """Transform args for the DINOv31 post-train method.

    ``record_geometry`` is on by default (the cross-view PaKA loss needs the crop
    boxes to ROI-align student locals with teacher globals). ``paka_num_local``
    high-overlap student local crops are sampled inside the globals at
    ``paka_high_overlap_iou`` containment. ``image_size`` is inherited from
    :class:`DINOv2ViTTransformArgs`; see the ``dinov31`` module docstring for the
    recommended post-train recipe.
    """

    record_geometry: bool = True
    paka_num_local: int = 8
    paka_high_overlap_iou: float = 0.9


class DINOv31Transform(DINOv2ViTTransform):
    """DINOv2 multi-crop transform extended with PaKA clean globals + locals.

    Produces the view layout the DINOv31 method indexes by position::

        [global0, global1, dino_local0..L-1,
         clean_global0, clean_global1, paka_local0..K-1]

    The two trailing clean globals are augmentation-free renders of the two global
    crops (same crop box / flips) and feed the PaKA clean teacher; the K trailing
    paka_locals are small crops sampled inside alternating globals (containment
    >= paka_high_overlap_iou) and feed the PaKA student. DINO/iBOT see only the
    leading ``[global0, global1, dino_local..]`` views unchanged.
    """

    def __init__(self, transform_args: DINOv31TransformArgs) -> None:
        super().__init__(transform_args=transform_args)
        # Stored for the clean-teacher / high-overlap-local renders in __call__.
        self._image_size = transform_args.image_size
        self._normalize = transform_args.normalize
        self._paka_num_local = transform_args.paka_num_local
        self._paka_high_overlap_iou = transform_args.paka_high_overlap_iou
        # The paka locals reuse the local view size and photometric config.
        assert transform_args.local_view is not None  # DINOv2ViT default is set.
        local_view = transform_args.local_view
        self._paka_local_size = local_view.view_size
        self._paka_local_gaussian_blur = local_view.gaussian_blur
        self._paka_color_jitter = transform_args.color_jitter
        self._paka_random_gray_scale = transform_args.random_gray_scale
        self._paka_solarize = transform_args.solarize
        self._paka_hflip_prob = (
            transform_args.random_flip.horizontal_prob
            if transform_args.random_flip is not None
            else 0.0
        )
        self._paka_vflip_prob = (
            transform_args.random_flip.vertical_prob
            if transform_args.random_flip is not None
            else 0.0
        )

    def __call__(self, input: TransformInput) -> TransformOutput:
        views = super().__call__(input)  # [global0, global1, dino_local..] + geometry

        # Two clean (augmentation-free) renders of the global crops, re-using
        # their recorded geometry, for the PaKA clean teacher. Appended at the
        # end so the leading [global, local] order DINO/iBOT rely on is preserved.
        for g in range(2):
            views.append(
                render_clean_global(
                    input=input,
                    geometry=views[g]["geometry"],
                    size=self._image_size,
                    normalize=self._normalize,
                )
            )

        # PaKA high-overlap student locals, appended last. Each is sampled inside
        # an alternating global crop so it overlaps heavily with the clean global
        # teacher that pairs with it (parent = local_idx % 2).
        if self._paka_num_local > 0:
            rng = random.Random()
            for local_idx in range(self._paka_num_local):
                parent_geom = views[local_idx % 2]["geometry"]
                parent_box = (
                    int(round(float(parent_geom[0]))),
                    int(round(float(parent_geom[1]))),
                    int(round(float(parent_geom[2]))),
                    int(round(float(parent_geom[3]))),
                )
                box = sample_high_overlap_box(
                    parent_box=parent_box,
                    min_iou=self._paka_high_overlap_iou,
                    rng=rng,
                )
                hflip = rng.random() < self._paka_hflip_prob
                vflip = rng.random() < self._paka_vflip_prob
                views.append(
                    render_augmented_local(
                        input=input,
                        box=box,
                        size=self._paka_local_size,
                        hflip=hflip,
                        vflip=vflip,
                        color_jitter=self._paka_color_jitter,
                        random_gray_scale=self._paka_random_gray_scale,
                        gaussian_blur=self._paka_local_gaussian_blur,
                        solarize=self._paka_solarize,
                        normalize=self._normalize,
                    )
                )
        return views

    @staticmethod
    def transform_args_cls() -> type[DINOv31TransformArgs]:
        return DINOv31TransformArgs
