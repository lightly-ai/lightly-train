#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Identity, Module, Sequential, Upsample

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)

if TYPE_CHECKING:
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import SPPF, C2f
    from ultralytics.nn.modules.conv import Conv

logger = logging.getLogger(__name__)


def _get_direct_url() -> str | None:
    """Get direct_url.json text from the installed ultralytics package.

    Returns:
        str | None: The source URL in the direct_url.json file or None if the package is installed from normal sources.
    """
    try:
        from importlib.metadata import Distribution, PackageNotFoundError

        dist = Distribution.from_name("ultralytics")
        return dist.read_text("direct_url.json")
    except PackageNotFoundError:
        return None


YOLOV12_ULTRALYTICS_AVAILABLE = RequirementCache("ultralytics>=8.3.78")
YOLOV12_ORIGINAL_AVAILABLE = _get_direct_url() is not None
YOLOV11_AVAILABLE = RequirementCache("ultralytics>=8.3.0")


class UltralyticsModelWrapper(Module, ModelWrapper):
    def __init__(self, model: YOLO) -> None:
        super().__init__()
        _enable_gradients(model=model)
        # Set model to training mode. This is necessary for Ultralytics pretrained
        # models as they are loaded in eval mode by default.
        model.model.train()
        self._model = [model]
        self._backbone, self._feature_dim = _get_backbone(model)
        self._pool = AdaptiveAvgPool2d((1, 1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self._backbone(x)}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> YOLO:
        return self._model[0]


def _get_backbone(model: YOLO) -> tuple[Sequential, int]:
    """Extracts the backbone and feature dimension from the model.

    Ultralytics doesn't provide a way to get the backbone of the model. All layers
    are stored in a Sequential object. We create the backbone by finding the last
    layer of the backbone based on model architecture.
    """

    from ultralytics.nn.modules.block import C3, SPPF, C2f
    from ultralytics.nn.modules.head import Classify

    # Ultralytics stores the actual model in YOLO.model.model
    seq = model.model.model
    assert isinstance(seq, Sequential)

    for module_idx, module in enumerate(seq):
        if module_idx == 0:
            continue

        last_module = seq[module_idx - 1]
        # Only keep the layers before SPPF for the backbone.
        # NOTE(Philipp, 11/24): Keep batch norm and activation for now. Our experiments
        # have shown that they improve performance of the pretrained model slightly.
        # We use type() instead of isinstance() to match the exact class and not subclasses.
        if (
            type(last_module) is C3 and type(module) is SPPF
        ):  # YOLOv5 ends with C3 before SPPF
            backbone = seq[:module_idx]
            feature_dim = last_module.cv3.conv.out_channels  # C3 block
            return backbone, feature_dim

        if (
            type(last_module) is Sequential and type(module) is SPPF
        ):  # YOLOv6 ends with a Sequential of Conv blocks before SPPF
            backbone = seq[:module_idx]
            feature_dim = last_module[-1].conv.out_channels  # Sequential of Conv blocks
            return backbone, feature_dim

        if type(last_module) is C2f and type(module) in (
            SPPF,
            Classify,
        ):  # YOLOv8 ends with C2f before SPPF or Classify
            backbone = seq[:module_idx]
            feature_dim = last_module.cv2.conv.out_channels  # C2f block
            return backbone, feature_dim

        if YOLOV11_AVAILABLE:
            from ultralytics.nn.modules.block import C2PSA

            if type(last_module) is C2PSA and type(module) in (
                Upsample,
                Classify,
            ):  # YOLOv11 ends with C2PSA before Upsample or Classify
                backbone = seq[:module_idx]
                feature_dim = last_module.cv2.conv.out_channels  # C2PSA block
                return backbone, feature_dim

        if YOLOV12_ULTRALYTICS_AVAILABLE or YOLOV12_ORIGINAL_AVAILABLE:
            from ultralytics.nn.modules.block import A2C2f

            if type(last_module) is A2C2f and type(module) in (
                Upsample,
                Classify,
            ):  # YOLOv12 ends with A2C2f before Upsample or Classify
                backbone = seq[:module_idx]
                feature_dim = last_module.cv2.conv.out_channels  # A2C2f block
                return backbone, feature_dim

    raise RuntimeError(f"Could not identify backbone in model {type(model)}")


def _sppf_skip_cv2_bn_act(sppf: SPPF) -> SPPF:
    """Creates a copy of the SPPF module with the last batchnorm and activation skipped.

    The unmodified weights are shared between the original and the new module!
    """
    new_sppf = copy.deepcopy(sppf)
    new_sppf.cv1 = sppf.cv1  # Keep cv1
    new_sppf.cv2 = _cv2_skip_bn_act(cv2=sppf.cv2)  # Replace bn and act in cv2
    new_sppf.m = sppf.m  # Keep m
    return new_sppf


def _c2f_skip_cv2_bn_act(c2f: C2f) -> C2f:
    """Creates a copy of the C2f module with the last batchnorm and activation skipped.

    The unmodified weights are shared between the original and the new module!
    """
    new_c2f = copy.deepcopy(c2f)
    new_c2f.cv1 = c2f.cv1  # Keep cv1
    new_c2f.cv2 = _cv2_skip_bn_act(cv2=c2f.cv2)  # Replace bn and act in cv2
    new_c2f.m = c2f.m  # Keep m
    return new_c2f


def _cv2_skip_bn_act(cv2: Conv) -> Conv:
    """Creates a copy of the Conv module with batchnorm and activation skipped.

    The unmodified weights are shared between the original and the new module!
    """
    new_cv2 = copy.deepcopy(cv2)
    new_cv2.conv = cv2.conv  # Keep the convolutional layer
    new_cv2.bn = Identity()  # Replace batchnorm with identity
    new_cv2.act = Identity()  # Replace activation with identity
    return new_cv2


def _enable_gradients(model: YOLO) -> None:
    """Enables gradients for parameters in the model.

    Ultralytics disables by default gradients for models loaded from a checkpoint
    and enables gradients for models loaded from a config file. This function enables
    gradients for training. It respects the 'freeze' argument in the model's config
    if it exists.
    """
    # The logic of the function follows the one in the original Ultralytics code:
    # https://github.com/ultralytics/ultralytics/blob/6dcc4a0610bf445212253fb51b24e29429a2bcc3/ultralytics/engine/trainer.py#L238C11-L258
    freeze_names = [".dfl"]
    if model.model is not None and model.model.args is not None:
        freeze = model.model.args.get("freeze")
        freeze_list = (
            freeze
            if isinstance(freeze, list)
            else range(freeze)
            if isinstance(freeze, int)
            else []
        )
        freeze_names.extend([f"model.{x}" for x in freeze_list])
    logger.debug(f"Freezing parameters with names {freeze_names}")
    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in freeze_names):
            logger.info(f"Disabling gradients for parameter '{name}'")
            param.requires_grad = False
        elif not param.requires_grad:
            logger.info(f"Enabling gradients for parameter '{name}'")
            param.requires_grad = True
