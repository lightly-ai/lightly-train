#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Disable CV2 threads to avoid slowdowns when used inside a dataloader with many workers.
# For details, see https://github.com/albumentations-team/albumentations/issues/1246
# cv2 is not added to the pyproject.toml dependencies to not require a specific
# distribution. However albumentations requires it, thus making sure that at least
# one distribution is available.
# See https://github.com/albumentations-team/albumentations/blob/e3b47b3a127f92541cfeb16abbb44a6f8bf79cc8/setup.py#L11C1-L17C1
import cv2

cv2.setNumThreads(0)

# Disable beta transforms warning by torchvision.
# See https://stackoverflow.com/questions/77279407
# TODO(Philipp, 09/24): Remove this once the warning is removed.
import torchvision

torchvision.disable_beta_transforms_warning()

# Disable albumentations update check.
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from lightly_train._commands.common_helpers import ModelFormat, ModelPart
from lightly_train._commands.embed import embed
from lightly_train._commands.export import export
from lightly_train._commands.export_task import export_onnx
from lightly_train._commands.train import train
from lightly_train._commands.train_task import train_semantic_segmentation
from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._methods.method_helpers import list_methods
from lightly_train._models.package_helpers import list_model_names as list_models
from lightly_train._task_models.task_model_helpers import load_model_from_checkpoint

__all__ = [
    "embed",
    "EmbeddingFormat",
    "export",
    "export_onnx",
    "list_methods",
    "list_models",
    "ModelFormat",
    "ModelPart",
    "train",
    "train_semantic_segmentation",
    "load_model_from_checkpoint",
]

__version__ = "0.11.1"
