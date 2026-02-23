#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train.types import NDArrayImage, NDArrayOBBoxes


class OrientedObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NotRequired[NDArrayOBBoxes]
    class_labels: NotRequired[NDArray[np.int64]]


class OrientedObjectDetectionTransformOutput(TaskTransformOutput):
    image: Tensor
    bboxes: NotRequired[Tensor]
    class_labels: NotRequired[Tensor]


class OrientedObjectDetectionTransformArgs(ObjectDetectionTransformArgs):
    pass


class ChannelDropTV(v2.Transform):
    def __init__(
        self,
        num_channels_keep: int,
        weight_drop: Sequence[float],
    ) -> None:
        super().__init__()
        self.num_channels_keep = num_channels_keep
        self.weight_drop = list(weight_drop)

        if num_channels_keep < 1:
            raise ValueError(
                f"num_channels_keep must be at least 1, got {num_channels_keep}."
            )
        if any(w < 0 for w in self.weight_drop):
            raise ValueError(
                f"All weights in weight_drop must be non-negative, got {self.weight_drop}."
            )
        if sum(w == 0 for w in self.weight_drop) > self.num_channels_keep:
            raise ValueError(
                "At most num_channels_keep channels can have zero weight "
                f"to guarantee they can be kept, got {self.num_channels_keep} and "
                f"{self.weight_drop}."
            )

        weight_array = torch.tensor(self.weight_drop, dtype=torch.float32)
        self._prob_drop = weight_array / weight_array.sum()

    def _get_params(self, flat_inputs: list) -> dict[str, Any]:
        num_channels = flat_inputs[0].shape[0]
        if self.num_channels_keep == num_channels:
            return {"channels_to_keep": torch.arange(num_channels)}

        channels_to_drop = torch.multinomial(
            self._prob_drop,
            num_samples=num_channels - self.num_channels_keep,
            replacement=False,
        )
        all_channels = torch.arange(num_channels)
        mask = torch.isin(all_channels, channels_to_drop)
        channels_to_keep = torch.sort(all_channels[~mask])[0]
        return {"channels_to_keep": channels_to_keep}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, Image):
            if inpt.ndim < 3:
                return inpt
            channels_to_keep = params["channels_to_keep"]
            if inpt.shape[0] != len(self.weight_drop):
                return inpt
            return Image(torch.index_select(inpt, 0, channels_to_keep))
        return inpt


class RandomRotate90OBB(v2.Transform):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def _get_params(self, flat_inputs: list) -> dict[str, Any]:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(1, 4, (1,)).item())
        else:
            k = 0
        return {"k": k}

    def _transform_image(self, img: Tensor, k: int) -> Tensor:
        if k == 0:
            return img
        return torch.rot90(img, k=k, dims=(-2, -1))

    def _transform_bboxes(
        self, bboxes: BoundingBoxes, k: int, h: int, w: int
    ) -> BoundingBoxes:
        if k == 0:
            return bboxes

        cx, cy, bw, bh, angle = (
            bboxes[:, 0],
            bboxes[:, 1],
            bboxes[:, 2],
            bboxes[:, 3],
            bboxes[:, 4],
        )
        angle_rad = angle * (3.141592653589793 / 180.0)

        if k == 1:
            new_cx = cy
            new_cy = w - cx
            new_bw = bh
            new_bh = bw
            new_angle_rad = angle_rad + (3.141592653589793 / 2.0)
        elif k == 2:
            new_cx = w - cx
            new_cy = h - cy
            new_bw = bw
            new_bh = bh
            new_angle_rad = angle_rad + 3.141592653589793
        else:
            new_cx = h - cy
            new_cy = cx
            new_bw = bh
            new_bh = bw
            new_angle_rad = angle_rad - (3.141592653589793 / 2.0)

        new_angle = (new_angle_rad * 180.0 / 3.141592653589793) % 360.0

        new_bboxes = torch.stack([new_cx, new_cy, new_bw, new_bh, new_angle], dim=1)
        new_h, new_w = (w, h) if k in (1, 3) else (h, w)
        return BoundingBoxes(  # type: ignore[call-arg]
            new_bboxes,
            format=BoundingBoxFormat.CXCYWHR,
            canvas_size=(new_h, new_w),
        )

    def forward(self, *inputs: Tensor) -> Any:
        params = self._get_params(list(inputs))
        k = params["k"]

        if k == 0 or len(inputs) == 0:
            return inputs if len(inputs) > 1 else (inputs[0] if inputs else inputs)

        outputs = []
        h, w = inputs[0].shape[-2:]

        for inpt in inputs:
            if isinstance(inpt, Image):
                outputs.append(Image(self._transform_image(inpt, k)))
            elif isinstance(inpt, BoundingBoxes):
                outputs.append(self._transform_bboxes(inpt, k, h, w))
            else:
                outputs.append(inpt)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]


class OrientedObjectDetectionTransform(TaskTransform):
    transform_args_cls: type[OrientedObjectDetectionTransformArgs] = (
        OrientedObjectDetectionTransformArgs
    )

    def __init__(
        self,
        transform_args: OrientedObjectDetectionTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        self.transform_args: OrientedObjectDetectionTransformArgs = transform_args
        self.stop_step = (
            transform_args.stop_policy.stop_step if transform_args.stop_policy else None
        )

        if self.stop_step is not None:
            raise NotImplementedError(
                "Stopping certain augmentations after some steps is not implemented yet."
            )
        self.global_step = 0
        self.stop_ops = (
            transform_args.stop_policy.ops if transform_args.stop_policy else set()
        )
        self.past_stop = False

        transforms_list: list[v2.Transform] = []

        if transform_args.channel_drop is not None:
            transforms_list.append(
                ChannelDropTV(
                    num_channels_keep=transform_args.channel_drop.num_channels_keep,
                    weight_drop=transform_args.channel_drop.weight_drop,
                )
            )

        if transform_args.photometric_distort is not None:
            transforms_list.append(
                v2.RandomPhotometricDistort(
                    brightness=transform_args.photometric_distort.brightness,
                    contrast=transform_args.photometric_distort.contrast,
                    saturation=transform_args.photometric_distort.saturation,
                    hue=transform_args.photometric_distort.hue,
                    p=transform_args.photometric_distort.prob,
                )
            )

        if transform_args.random_zoom_out is not None:
            transforms_list.append(
                v2.RandomZoomOut(
                    fill=transform_args.random_zoom_out.fill,
                    side_range=transform_args.random_zoom_out.side_range,
                    p=transform_args.random_zoom_out.prob,
                )
            )

        if transform_args.random_iou_crop is not None:
            raise NotImplementedError(
                "RandomIoUCrop is not implemented yet for OrientedObjectDetectionTransform."
                "torchvision does not support it for now."
            )

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                transforms_list.append(
                    v2.RandomHorizontalFlip(
                        p=transform_args.random_flip.horizontal_prob
                    )
                )
            if transform_args.random_flip.vertical_prob > 0.0:
                transforms_list.append(
                    v2.RandomVerticalFlip(p=transform_args.random_flip.vertical_prob)
                )

        if transform_args.random_rotate_90 is not None:
            transforms_list.append(
                RandomRotate90OBB(p=transform_args.random_rotate_90.prob)
            )

        if transform_args.random_rotate is not None:
            transforms_list.append(
                v2.RandomApply(
                    [
                        v2.RandomRotation(
                            degrees=transform_args.random_rotate.degrees,  # type: ignore[arg-type]
                            interpolation=v2.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=transform_args.random_rotate.prob,
                )
            )

        if transform_args.resize is not None:
            transforms_list.append(
                v2.Resize(
                    size=(
                        no_auto(transform_args.resize.height),
                        no_auto(transform_args.resize.width),
                    ),
                    antialias=True,
                )
            )

        transforms_list.append(v2.ToDtype(torch.float32, scale=True))

        if transform_args.normalize is not None:
            normalize_args = no_auto(transform_args.normalize)
            transforms_list.append(
                v2.Normalize(
                    mean=list(normalize_args.mean),
                    std=list(normalize_args.std),
                )
            )

        self.transform = v2.Compose(transforms_list)

    def __call__(
        self, input: OrientedObjectDetectionTransformInput
    ) -> OrientedObjectDetectionTransformOutput:
        if (
            self.stop_step is not None
            and self.global_step >= self.stop_step
            and not self.past_stop
        ):
            raise NotImplementedError("Stop policy is not implemented yet.")

        assert "bboxes" in input, (
            "Input must contain bboxes for oriented object detection transform."
        )
        assert "class_labels" in input, (
            "Input must contain class_labels for oriented object detection transform."
        )

        image_hwc = input["image"]
        bboxes_np = input["bboxes"]
        class_labels = input["class_labels"]

        h, w = image_hwc.shape[:2]
        image_chw = np.transpose(image_hwc, (2, 0, 1))
        tv_image = Image(torch.from_numpy(image_chw))
        tv_bboxes = BoundingBoxes(  # type: ignore[call-arg]
            torch.from_numpy(bboxes_np),
            format=BoundingBoxFormat.CXCYWHR,
            canvas_size=(h, w),
        )

        transformed_image, transformed_bboxes = self.transform(tv_image, tv_bboxes)

        return {
            "image": transformed_image,
            "bboxes": transformed_bboxes,
            "class_labels": torch.from_numpy(class_labels),
        }
