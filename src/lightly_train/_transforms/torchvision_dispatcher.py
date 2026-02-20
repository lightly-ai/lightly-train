from typing import Mapping

import cv2
import numpy as np
from albumentations import DualTransform
from numpy.typing import NDArray
from torchvision.transforms import InterpolationMode, v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

from lightly_train.types import NDArrayImage, NDArrayOBBoxes


def numpy_image_to_tv_tensor_image(image_hwc: NDArrayImage) -> Image:
    """
    Convert a numpy image array to a torchvision tv_tensor Image.

    Args:
        image_hwc: A numpy array of shape (H, W, C) containing the image data.
    Returns:
        A torchvision tv_tensor Image containing the image data.
    """
    image_chw = image_hwc.transpose(2, 0, 1)

    return Image(image_chw)


def numpy_obb_to_tv_tensor_obb(
    oriented_bboxes: NDArrayOBBoxes, canvas_size: tuple[int, int]
) -> BoundingBoxes:
    """
    Convert oriented bounding boxes in the format (x_center, y_center, width, height, angle)
    to a torchvision tv_tensor BoundingBoxes object with format CXCYWHR.

    Args:
        oriented_bboxes: A numpy array of shape (n_boxes, 5) containing the oriented bounding boxes in (x_center, y_center, width, height, angle) format.
    Returns:
        A torchvision tv_tensor BoundingBoxes object containing the bounding boxes in XYWHR format.

    """
    return BoundingBoxes(
        oriented_bboxes,
        format=BoundingBoxFormat.CXCYWHR,
        canvas_size=canvas_size,
    )


def image_hwc_height_width(image: NDArrayImage) -> tuple[int, int]:
    """
    Get the height and width of an image from its shape.

    Args:
        image: A numpy array of shape (H, W, C) containing the image data.
    Returns:
        A tuple (height, width) containing the height and width of the image.
    """
    return image.shape[0], image.shape[1]


def convert_numpy_to_torchvision_input(
    image: NDArrayImage,
    bboxes: NDArrayOBBoxes | None,
) -> tuple[Image, BoundingBoxes | None]:
    tv_image = numpy_image_to_tv_tensor_image(image)
    tv_bboxes = (
        numpy_obb_to_tv_tensor_obb(bboxes, canvas_size=image_hwc_height_width(image))
        if bboxes is not None
        else None
    )
    return tv_image, tv_bboxes


def convert_torchvision_to_numpy_output(
    tv_image: Image,
    tv_bboxes: BoundingBoxes | None,
) -> tuple[NDArrayImage, NDArrayOBBoxes | None]:
    image_chw = tv_image.cpu().numpy()
    image_hwc = image_chw.transpose(1, 2, 0)
    bboxes = tv_bboxes.cpu().numpy() if tv_bboxes is not None else None
    return image_hwc, bboxes


class TorchVisionTransformDispatcher(DualTransform):
    def __init__(self, transform: v2.Transform, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transform = transform

    def __call__(
        self,
        image: NDArrayImage,
        oriented_bboxes: NDArrayOBBoxes | None = None,
        class_labels: NDArray[np.float64] | None = None,
        **kwargs,
    ) -> dict:
        tv_image, tv_bboxes = convert_numpy_to_torchvision_input(image, oriented_bboxes)
        tv_image_out, tv_bboxes_out = self.transform(tv_image, tv_bboxes)
        out_image, out_bboxes = convert_torchvision_to_numpy_output(
            tv_image_out, tv_bboxes_out
        )
        return {
            "image": out_image,
            "oriented_bboxes": out_bboxes,
            "class_labels": class_labels,
        }


CV2_INTERPOLATION_MODE = int

CV2_TO_TV_INTERPOLATION: Mapping[CV2_INTERPOLATION_MODE, InterpolationMode] = {
    cv2.INTER_NEAREST: InterpolationMode.NEAREST,
    cv2.INTER_LINEAR: InterpolationMode.BILINEAR,
    cv2.INTER_CUBIC: InterpolationMode.BICUBIC,
    cv2.INTER_AREA: InterpolationMode.BOX,
    cv2.INTER_LANCZOS4: InterpolationMode.LANCZOS,
}


def _get_interpolation_mode(
    cv2_interpolation: CV2_INTERPOLATION_MODE,
) -> InterpolationMode:
    return CV2_TO_TV_INTERPOLATION[cv2_interpolation]


TORCHVISION_ROTATE_SUPPORTED_INTERPOLATION_MODES = {
    InterpolationMode.NEAREST,
    InterpolationMode.BILINEAR,
}


def _check_supported_interpolation_mode_for_rotate(
    interpolation: InterpolationMode,
) -> None:
    if interpolation not in TORCHVISION_ROTATE_SUPPORTED_INTERPOLATION_MODES:
        raise ValueError(
            f"Unsupported interpolation mode {interpolation} for rotation. "
            f"Supported modes are: {TORCHVISION_ROTATE_SUPPORTED_INTERPOLATION_MODES}"
        )


class TorchVisionRotate90(TorchVisionTransformDispatcher):
    def __init__(self, p: float = 1.0) -> None:
        transform = v2.RandomChoice(
            [
                v2.RandomRotation(degrees=[0, 0]),
                v2.RandomRotation(degrees=[90, 90], expand=True),
                v2.RandomRotation(degrees=[180, 180]),
                v2.RandomRotation(degrees=[270, 270], expand=True),
            ]
        )

        super().__init__(
            transform=transform,
            p=p,
        )


class TorchVisionRotate(TorchVisionTransformDispatcher):
    def __init__(
        self,
        degrees: float | tuple[float, float],
        interpolation: CV2_INTERPOLATION_MODE,
        p: float = 1.0,
    ) -> None:
        tv_interpolation = _get_interpolation_mode(interpolation)
        _check_supported_interpolation_mode_for_rotate(tv_interpolation)

        super().__init__(
            transform=v2.RandomRotation(
                degrees=degrees, interpolation=tv_interpolation
            ),
            p=p,
        )


class TorchVisionHorizontalFlip(TorchVisionTransformDispatcher):
    def __init__(self, p: float) -> None:
        super().__init__(v2.RandomHorizontalFlip(p=1))


class TorchVisionVerticalFlip(TorchVisionTransformDispatcher):
    def __init__(self, p: float) -> None:
        super().__init__(v2.RandomVerticalFlip(p=1))


class TorchVisionResize(TorchVisionTransformDispatcher):
    def __init__(
        self,
        height: int,
        width: int,
        interpolation_mode: CV2_INTERPOLATION_MODE = cv2.INTER_LINEAR,
    ) -> None:
        super().__init__(
            v2.Resize(
                size=(height, width),
                interpolation=CV2_TO_TV_INTERPOLATION[interpolation_mode],
            )
        )


class TorchVisionRandomIoUCrop(TorchVisionTransformDispatcher):
    def __init__(
        self,
        min_scale,
        max_scale,
        min_aspect_ratio,
        max_aspect_ratio,
        sampler_options,
        crop_trials,
        iou_trials,
        p: float,
    ) -> None:
        super().__init__(
            v2.RandomIoUCrop(
                min_scale=min_scale,
                max_scale=max_scale,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                sampler_options=sampler_options,
                trials=iou_trials,
            ),
            p=p,
        )


class TorchVisionScaleJitter(TorchVisionTransformDispatcher):
    def __init__(
        self,
        target_size: tuple[int, int],
        scale_range: tuple[float, float] | None,
        p: float,
    ) -> None:
        scale_range = scale_range if scale_range is not None else (0.1, 2.0)

        super().__init__(
            v2.ScaleJitter(
                target_size=target_size,
                scale_range=scale_range,
            ),
            p=p,
        )
