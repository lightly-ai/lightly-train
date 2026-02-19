import numpy as np
import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2


@pytest.fixture
def dummy_np_image() -> np.ndarray:
    return np.random.rand(100, 100, 3).astype(np.float32)


@pytest.fixture
def dummy_np_obb() -> np.ndarray:
    # 5 values per box: (x_center, y_center, width, height, angle)
    return np.array([[50.0, 50.0, 20.0, 10.0, 45.0]], dtype=np.float32)


def test_numpy_image_to_tv_tensor_image_sucess(dummy_np_image: np.ndarray) -> None:
    from lightly_train._transforms.torchvision_dispatcher import (
        numpy_image_to_tv_tensor_image,
    )

    tv_image = numpy_image_to_tv_tensor_image(dummy_np_image)
    assert tv_image.shape == (3, 100, 100)
    assert tv_image.dtype == torch.float32
    assert isinstance(tv_image, tv_tensors.Image)


def test_numpy_obb_to_tv_tensor_obb_success(dummy_np_obb: np.ndarray) -> None:
    from lightly_train._transforms.torchvision_dispatcher import (
        numpy_obb_to_tv_tensor_obb,
    )

    canvas_size = (100, 100)
    tv_bboxes = numpy_obb_to_tv_tensor_obb(dummy_np_obb, canvas_size=canvas_size)
    assert tv_bboxes.shape == (1, 5)
    assert tv_bboxes.dtype == torch.float32
    assert isinstance(tv_bboxes, tv_tensors.BoundingBoxes)


class MockTransform(v2.Transform):
    def __call__(self, *args):
        return tuple(args)


def test_vision_dispatcher_call_returns_same(
    dummy_np_obb: np.ndarray, dummy_np_image: np.ndarray
):
    mock_transform = MockTransform()

    from lightly_train._transforms.torchvision_dispatcher import TorchVisionDispatcher

    dispatcher = TorchVisionDispatcher(transform=mock_transform)

    output = dispatcher(image=dummy_np_image, bboxes=dummy_np_obb)
    assert isinstance(output, dict)
    assert "image" in output
    assert "bboxes" in output
    assert np.array_equal(output["image"], dummy_np_image)
    assert np.array_equal(output["bboxes"], dummy_np_obb)


def test_vision_dispatcher_correctly_modifies(
    dummy_np_obb: np.ndarray, dummy_np_image: np.ndarray
):
    from lightly_train._transforms.torchvision_dispatcher import TorchVisionDispatcher

    dispatcher = TorchVisionDispatcher(transform=v2.Resize(size=(50, 50)))

    output = dispatcher(image=dummy_np_image, bboxes=dummy_np_obb)
    breakpoint()
    assert isinstance(output, dict)
    assert "image" in output
    assert output["image"].shape[0] == 50
    assert output["image"].shape[1] == 50
    assert output["image"].shape[2] == 3
    assert "bboxes" in output
    assert not np.array_equal(output["bboxes"], dummy_np_obb)
