#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
from typing import Callable

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Module

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)

logger = logging.getLogger(__name__)


class TIMMModelWrapper(Module, ModelWrapper):
    def __init__(self, model: Module) -> None:
        if not hasattr(model, "forward_features"):
            raise ValueError("Model must have a 'forward_features' method")
        if not hasattr(model, "num_features"):
            raise ValueError("Model must have a 'num_features' attribute")
        super().__init__()

        # TODO: It would be better to not save the full model but only the necessary
        # modules to calculate features. This would save memory and make sure we only
        # train the necessary parameters. Saving all parameters also requires us to
        # use `ddp_find_unused_parameters=True` in the Trainer.
        self._model = model
        self._pool = _get_pool_layer(model=model)
        self._forward_features = _get_forward_features_fn(model=model)

    def feature_dim(self) -> int:
        num_features: int = self._model.num_features
        return num_features

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        features = self._forward_features(self._model, x)
        return {"features": features}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        features = self._pool(x["features"])
        while len(features.shape) < 4:
            features = features.unsqueeze(-1)
        return {"pooled_features": features}

    def get_model(self) -> Module:
        return self._model


def _get_forward_features_fn(model: Module) -> Callable[[Module, Tensor], Tensor]:
    """Get the forward_features function for the model.

    Timm defines a model.forward_features method for all models, but the outputs are
    not always in NCHW format. Transformer models often return tensors in NLC shape,
    including the class and prefix tokens.
    Newer timm versions (>1.0) include a forward_intermediates method for some models,
    which allows us to get the last layer features consistently in NCHW format. We use
    this method if available, otherwise we use the forward_features method.
    """
    if hasattr(model, "forward_intermediates"):
        # For example VisionTransformer:
        # https://github.com/huggingface/pytorch-image-models/blob/e748805be31318da1a0e34b61294704666f50397/timm/models/vision_transformer.py#L635
        return _forward_intermediates
    elif hasattr(model, "get_intermediate_layers"):
        # Older versions of timm  (<1.0, >=0.9) use get_intermediate_layers instead of
        # forward_intermediates. See:
        # https://github.com/huggingface/pytorch-image-models/blob/e748805be31318da1a0e34b61294704666f50397/timm/models/vision_transformer.py#L717
        return _forward_get_intermediate_layers
    else:
        return _forward_features


def _forward_features(model: Module, x: Tensor) -> Tensor:
    x = model.forward_features(x)
    x = _drop_prefix_tokens(model, x)
    x = _to_nchw(x)
    return x


def _forward_get_intermediate_layers(model: Module, x: Tensor) -> Tensor:
    intermediates: Tensor = model.get_intermediate_layers(
        x,
        n=1,  # Only return the n=1 last layers.
        reshape=True,  # Reshape the output to NCHW format.
        norm=True,  # Apply normalization to be consistent with forward_features.
    )
    return intermediates[0]


def _forward_intermediates(model: Module, x: Tensor) -> Tensor:
    intermediates: Tensor = model.forward_intermediates(
        x,
        indices=1,  # Only return the indices=1 last layers.
        output_fmt="NCHW",
        intermediates_only=True,
        norm=True,  # Apply normalization to be consistent with forward_features.
    )
    return intermediates[0]


def _get_pool_layer(model: Module) -> Module:
    """Get the pooling layer from the model.

    Sadly, timm doesn't have a consistent way of storing the pooling layer.
    This function tries to find the pooling layer in the model. If it can't find it, it
    defaults to AdaptiveAvgPool2d.
    """
    if hasattr(model, "global_pool") and callable(model.global_pool):
        # Get global_pool stored on the model. For example for ResNet:
        # https://github.com/huggingface/pytorch-image-models/blob/e748805be31318da1a0e34b61294704666f50397/timm/models/resnet.py#L526
        global_pool: Module = model.global_pool
        return global_pool
    if (
        hasattr(model, "head")
        and hasattr(model.head, "global_pool")
        and callable(model.head.global_pool)
    ):
        # Get global_pool stored on the head. For example for RegNet:
        # https://github.com/huggingface/pytorch-image-models/blob/e748805be31318da1a0e34b61294704666f50397/timm/models/regnet.py#L452
        global_pool_head: Module = model.head.global_pool
        return global_pool_head
    logger.warning(
        "Could not find pooling layer on the model, defaulting to AdaptiveAvgPool2d"
    )
    # Return default pooling layer. For example VisionTransformer has some hardcoded
    # logic in forward_head on how to pool features but this is not easily accessible:
    # https://github.com/huggingface/pytorch-image-models/blob/e748805be31318da1a0e34b61294704666f50397/timm/models/vision_transformer.py#L749-L754
    #
    # NOTE(Guarin, 05/24): In the future we could try using model.attn_pool if
    # available. But attn_pool usually expects NLC input and not NCHW, so we would have
    # to handle this accordingly. See:
    # https://github.com/huggingface/pytorch-image-models/blob/e748805be31318da1a0e34b61294704666f50397/timm/models/vision_transformer.py#L749
    return AdaptiveAvgPool2d((1, 1))


def _drop_prefix_tokens(model: Module, x: Tensor) -> Tensor:
    """Removes all prefix/class tokens from the tensor."""
    if len(x.shape) == 3:
        # Some models have a num_prefix_tokens attribute. See:
        # https://github.com/huggingface/pytorch-image-models/blob/832d3618a5f989dbd4f4388842f341c8352e7b0a/timm/models/vision_transformer.py#L472
        num_prefix_tokens = getattr(model, "num_prefix_tokens", None)
        # Some models only have a cls_token. See:
        # https://github.com/huggingface/pytorch-image-models/blob/832d3618a5f989dbd4f4388842f341c8352e7b0a/timm/models/xcit.py#L362
        if num_prefix_tokens is None:
            if hasattr(model, "cls_token"):
                num_prefix_tokens = 1
            else:
                num_prefix_tokens = 0
        return x[:, num_prefix_tokens:]
    # Assume no prefix tokens.
    return x


def _to_nchw(x: Tensor) -> Tensor:
    """Convert tensor to NCHW format."""
    if len(x.shape) == 3:
        N, L, C = x.shape
        # Assumes square input.
        # TODO: Handle non-square inputs. See:
        # https://github.com/huggingface/pytorch-image-models/blob/832d3618a5f989dbd4f4388842f341c8352e7b0a/timm/models/vision_transformer.py#L698
        H = W = int(L**0.5)
        return x.reshape(N, H, W, C).permute(0, 3, 1, 2).contiguous()
    # Assume NCHW format.
    return x
