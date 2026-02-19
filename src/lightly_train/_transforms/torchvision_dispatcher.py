from typing import Mapping

import cv2
from albumentations import DualTransform
from torchvision.transforms import InterpolationMode, v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

from lightly_train.types import NDArrayImage, NDArrayOBBoxes


def numpy_image_to_tv_tensor_image(image_hwc: NDArrayImage) -> Image:
    """
    Convert a numpy image array to a torchvision tv_tensor Image.

    Args:
        image: A numpy array of shape (H, W, C) containing the image data.
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


class TorchVisionDispatcher(DualTransform):
    def __init__(self, transform: v2.Transform, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.transform = transform

    def __call__(self, image, bboxes=None, class_labels=None, **kwargs) -> dict:
        tv_image, tv_bboxes = convert_numpy_to_torchvision_input(image, bboxes)
        tv_image_out, tv_bboxes_out = self.transform(tv_image, tv_bboxes)
        out_image, out_bboxes = convert_torchvision_to_numpy_output(
            tv_image_out, tv_bboxes_out
        )
        return {"image": out_image, "bboxes": out_bboxes, "classes": class_labels}


class TorchVisionRotate90(TorchVisionDispatcher):
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


CV2_INTERPOLATION_MODE = int

CV2_TO_TV_INTERPOLATION: Mapping[CV2_INTERPOLATION_MODE, InterpolationMode] = {
    cv2.INTER_NEAREST: InterpolationMode.NEAREST,
    cv2.INTER_LINEAR: InterpolationMode.BILINEAR,
    cv2.INTER_CUBIC: InterpolationMode.BICUBIC,
    cv2.INTER_AREA: InterpolationMode.BOX,
    cv2.INTER_LANCZOS4: InterpolationMode.LANCZOS,
}


class TorchVisionRotate(TorchVisionDispatcher):
    def __init__(
        self,
        degrees: float | tuple[float, float],
        interpolation: CV2_INTERPOLATION_MODE,
        p: float = 1.0,
    ) -> None:
        super().__init__(
            transform=v2.RandomRotation(
                degrees=degrees,
                interpolation=CV2_TO_TV_INTERPOLATION[interpolation],
            ),
            p=p,
        )


class TorchVisionHorizontalFlip(TorchVisionDispatcher):
    def __init__(self, p: float) -> None:
        super().__init__(v2.RandomHorizontalFlip(p=1))


class TorchVisionVerticalFlip(TorchVisionDispatcher):
    def __init__(self, p: float) -> None:
        super().__init__(v2.RandomVerticalFlip(p=1))


class TorchVisionResize(TorchVisionDispatcher):
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


class TorchVisionRandomIoUCrop(TorchVisionDispatcher):
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
