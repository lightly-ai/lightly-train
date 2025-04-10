#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import inspect
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import typing_extensions
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Conv2d, Module

from lightly_train._checkpoint import (
    Checkpoint,
    CheckpointLightlyTrain,
    CheckpointLightlyTrainModels,
)
from lightly_train._commands import extract_video_frames
from lightly_train._configs.config import PydanticConfig
from lightly_train._methods.method import Method
from lightly_train._methods.simclr.simclr import SimCLR, SimCLRArgs
from lightly_train._models import (
    package_helpers as feature_extractor_api,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.feature_extractor import FeatureExtractor
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._transforms.transform import (
    MethodTransform,
    NormalizeArgs,
)
from lightly_train.types import TransformInput, TransformOutput


class DummyCustomModel(Module):
    def __init__(self, feature_dim: int = 2):
        super().__init__()
        self._feature_dim = feature_dim
        self.conv = Conv2d(in_channels=3, out_channels=feature_dim, kernel_size=2)
        self.global_pool = AdaptiveAvgPool2d(output_size=(1, 1))

    def feature_dim(self) -> int:
        return self._feature_dim

    # Not typed as ForwardFeaturesOutput to have same interface as users.
    def forward_features(self, x: Tensor) -> dict[str, Any]:
        return {"features": self.conv(x)}

    # Not typed as ForwardFeaturesOutput -> ForwardPoolOutput to have same interface
    # as users.
    def forward_pool(self, x: dict[str, Any]) -> dict[str, Any]:
        return {"pooled_features": self.global_pool(x["features"])}

    def get_model(self) -> Module:
        return self.conv


class DummyMethodTransform(MethodTransform):
    def __init__(self) -> None:
        self.transform = ToTensorV2()

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [self.transform(**input)]


def get_model() -> Module:
    return DummyCustomModel()


def get_feature_extractor(model: Module | None = None) -> FeatureExtractor:
    if model is None:
        model = get_model()
    return feature_extractor_api.get_feature_extractor(model=model)


def get_embedding_model(model: Module | None = None) -> EmbeddingModel:
    return EmbeddingModel(feature_extractor=get_feature_extractor(model=model))


def get_method(model: Module | None = None) -> Method:
    return SimCLR(
        method_args=SimCLRArgs(),
        optimizer_args=AdamWArgs(),
        embedding_model=get_embedding_model(model=model),
        global_batch_size=2,
    )


def get_checkpoint(
    model: Module | None = None, dtype: torch.dtype = torch.float32
) -> Checkpoint:
    if model is None:
        model = get_model()
    embedding_model = get_embedding_model(model=model).to(dtype)
    method = get_method(model=model).to(dtype)
    return Checkpoint(
        state_dict=method.state_dict(),
        lightly_train=CheckpointLightlyTrain.from_now(
            models=CheckpointLightlyTrainModels(
                model=model, embedding_model=embedding_model
            ),
            normalize_args=NormalizeArgs(),
        ),
    )


def create_image(
    path: Path, height: int = 128, width: int = 128, mode: str = "RGB"
) -> None:
    img_np = np.random.uniform(0, 255, size=(width, height, 3))
    img = Image.fromarray(img_np.astype(np.uint8)).convert(mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_images(
    image_dir: Path,
    files: int | Iterable[str] = 10,
    height: int = 128,
    width: int = 128,
    mode: str = "RGB",
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(files, int):
        files = [f"{i}.png" for i in range(files)]
    for filename in files:
        create_image(path=image_dir / filename, height=height, width=width, mode=mode)


def create_video(video_path: Path, n_frames: int = 10) -> None:
    extract_video_frames.assert_ffmpeg_is_installed()
    frame_dir = video_path.parent / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    create_images(image_dir=frame_dir, files=n_frames)
    cmd = [
        "ffmpeg",
        "-framerate",
        "1",
        "-i",
        str(frame_dir / "%d.png"),
        "-c:v",
        "libx264",
        "-vf",
        "fps=1",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def create_videos(
    videos_dir: Path, n_videos: int = 4, n_frames_per_video: int = 10
) -> None:
    extract_video_frames.assert_ffmpeg_is_installed()
    videos_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_videos):
        create_video(
            video_path=videos_dir / f"video_{i}.mp4",
            n_frames=n_frames_per_video,
        )


def assert_same_params(
    a: type[PydanticConfig] | Callable,  # type: ignore[type-arg]
    b: type[PydanticConfig] | Callable,  # type: ignore[type-arg]
    assert_type: bool = True,
    assert_required: bool = True,
    assert_default: bool = True,
) -> None:
    """Assert that the parameters of a PydanticConfig and a function are the same."""

    @dataclass
    class ParamInfo:
        name: str
        required: bool
        default: Any
        type: Any

    def _get_config_params(config: type[PydanticConfig]) -> dict[str, ParamInfo]:
        fields = config.model_fields
        type_hints = typing_extensions.get_type_hints(config)
        return {
            name: ParamInfo(
                name=name,
                required=field.is_required(),
                default=field.get_default(),
                type=type_hints[name],
            )
            for name, field in fields.items()
        }

    def _get_fn_params(fn: Callable) -> dict[str, ParamInfo]:  # type: ignore[type-arg]
        signature = inspect.signature(fn)
        type_hints = typing_extensions.get_type_hints(fn)
        return {
            name: ParamInfo(
                name=name,
                required=param.default is inspect.Parameter.empty,
                default=param.default,
                type=type_hints[name],
            )
            for name, param in signature.parameters.items()
        }

    def _get_params(obj: type[PydanticConfig] | Any) -> dict[str, ParamInfo]:
        if inspect.isclass(obj) and issubclass(obj, PydanticConfig):
            return _get_config_params(obj)
        return _get_fn_params(obj)

    a_params = _get_params(a)
    b_params = _get_params(b)

    # Check that both have the same parameter names.
    assert a_params.keys() == b_params.keys()

    if assert_type:
        a_types = {a.name: a.type for a in a_params.values()}
        b_types = {b.name: b.type for b in b_params.values()}
        assert a_types == b_types
    if assert_required:
        a_required = {a.name for a in a_params.values() if a.required}
        b_required = {b.name for b in b_params.values() if b.required}
        assert a_required == b_required
    if assert_default:
        a_defaults = {a.name: a.default for a in a_params.values() if not a.required}
        b_defaults = {b.name: b.default for b in b_params.values() if not b.required}
        assert a_defaults == b_defaults
