#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from albumentations import BboxParams

from lightly_train._transforms.transform import (
    ChannelDropArgs,
    CopyBlendArgs,
    MixUpArgs,
    MosaicArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomIoUCropArgs,
    RandomPhotometricDistortArgs,
    RandomRotate90Args,
    RandomRotationArgs,
    RandomZoomOutArgs,
    ResizeArgs,
    ScaleJitterArgs,
)
from lightly_train.types import ImageSizeTuple


@runtime_checkable
class LTDETRTransformArgs(Protocol):
    """Structural contract for the shared LT-DETR sample/collate machinery.

    The step-scheduling helpers (:func:`build_ltdetr_sample_transform_parts`,
    :class:`StepScheduledCompose`, and the LT-DETR collate functions) read these
    fields off the transform arguments they are given. Concrete task args (e.g.
    ``LTDETRObjectDetectionTransformArgs``,
    ``LTDETRInstanceSegmentationTransformArgs``) are Pydantic models that
    structurally satisfy this protocol, so the shared machinery can stay agnostic
    to the concrete task.
    """

    @property
    def channel_drop(self) -> ChannelDropArgs | None: ...

    @property
    def num_channels(self) -> int | Literal["auto"]: ...

    @property
    def photometric_distort(self) -> RandomPhotometricDistortArgs | None: ...

    @property
    def random_zoom_out(self) -> RandomZoomOutArgs | None: ...

    @property
    def random_iou_crop(self) -> RandomIoUCropArgs | None: ...

    @property
    def random_flip(self) -> RandomFlipArgs | None: ...

    @property
    def random_rotate_90(self) -> RandomRotate90Args | None: ...

    @property
    def random_rotate(self) -> RandomRotationArgs | None: ...

    @property
    def image_size(self) -> ImageSizeTuple | Literal["auto"]: ...

    @property
    def mixup(self) -> MixUpArgs | None: ...

    @property
    def copyblend(self) -> CopyBlendArgs | None: ...

    @property
    def mosaic(self) -> MosaicArgs | None: ...

    @property
    def scale_jitter(self) -> ScaleJitterArgs | None: ...

    @property
    def resize(self) -> ResizeArgs | None: ...

    @property
    def bbox_params(self) -> BboxParams | None: ...

    @property
    def normalize(self) -> NormalizeArgs | Literal["auto"] | None: ...
