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
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)

logger = logging.getLogger(__name__)

# Architecture name prefix → ArchitectureInfo.
# Architecture names come from model.pretrained_cfg.architecture (lowercase).
_TIMM_ARCH_NAME_PREFIXES: list[tuple[str, ArchitectureInfo]] = [
    # Transformers
    ("vit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("deit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("deit3_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("swin_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("swinv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("beit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("beitv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("beit3_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("eva_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("eva02_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("xcit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("flexivit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("tiny_vit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("cait_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("crossvit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("pit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("tnt_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("twins_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("nest_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("sequencer2d_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("aimv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("convit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("gmixer_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("gmlp_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("hiera_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("hieradet_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("mixer_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("mvitv2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("naflexvit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("pvt_v2_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("resmlp_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("sam2_hiera_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("samvit_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("visformer_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("vitamin_", {"model_type": "transformer", "norm_type": "layernorm"}),
    ("volo_", {"model_type": "transformer", "norm_type": "layernorm"}),
    # Hybrids
    ("maxvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("coat_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("cvt_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("levit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("cmt_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("mix_transformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("caformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("coatnet_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("coatnext_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("davit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("edgenext_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("efficientformerv2_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("efficientformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("efficientvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("fastvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("gcvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("maxxvitv2_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("maxxvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("mobilevitv2_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("mobilevit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("nextvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("repvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("shvit_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    ("swiftformer_", {"model_type": "hybrid", "norm_type": "layernorm"}),
    # Hybrid + BatchNorm (attention augmented conv networks)
    ("bat_", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("botnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("eca_botnext", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("eca_halonext", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("halo", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("lambda_resnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("lamhalobotnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("sebotnet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    ("sehalonet", {"model_type": "hybrid", "norm_type": "batchnorm"}),
    # Conv + LayerNorm
    ("convnextv2_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("convnext_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("convmixer_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("poolformerv2_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("poolformer_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("convformer_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("fasternet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("focalnet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("inception_next_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("mambaout_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("rdnet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    ("starnet_", {"model_type": "convolutional", "norm_type": "layernorm"}),
    # Conv + BatchNorm
    ("resnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("resnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("resnest", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("seresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("seresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnetx_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnety_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnetv_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("regnetz_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("efficientnetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("efficientnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tf_efficientnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv5_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv4_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv3_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenetv1_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobilenet_edgetpu", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mobileone_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("shufflenet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("densenet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("dpn", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("nfnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_nfnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("dm_nfnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("nf_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("vgg", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("squeezenet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tresnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hardcorenas_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cs3", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("csat", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cspdarknet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cspresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("cspresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("darknet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("dla", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_resnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_resnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("eca_vovnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ecaresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ecaresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ese_vovnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("fbnetc_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("fbnetv3_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gc_efficientnetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gcresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gcresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("gernet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("ghostnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hgnetv2_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hgnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("hrnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("inception_resnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("inception_v", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("lcnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("legacy_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mixnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("mnasnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("nasnetalarge", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("pnasnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("repghostnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("repvgg_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("res2net", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("res2next", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("rexnetr_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("rexnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("sedarknet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("selecsls", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("semnasnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("senet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("skresnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("skresnext", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("spnasnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tf_mixnet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tf_mobilenetv3_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("tinynet_", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("vovnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("wide_resnet", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    ("xception", {"model_type": "convolutional", "norm_type": "batchnorm"}),
    # Test models (timm internal, mixed architectures)
    ("test_", {"model_type": "transformer", "norm_type": "layernorm"}),
]


class TIMMModelWrapper(Module, ModelWrapper, ArchitectureInfoGettable):
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
        num_features: int = self._model.num_features  # type: ignore[assignment]
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

    def architecture_info(self) -> ArchitectureInfo:
        arch_name = ""
        if hasattr(self._model, "pretrained_cfg") and isinstance(
            self._model.pretrained_cfg, dict
        ):
            arch_name = self._model.pretrained_cfg.get("architecture", "").lower()
        for prefix, arch_info in _TIMM_ARCH_NAME_PREFIXES:
            if arch_name.startswith(prefix):
                return arch_info
        logger.warning(
            "Could not infer architecture info from TIMM's pretrained_cfg, "
            "falling back to classifying as Transformer-based architecture."
        )
        return {"model_type": "transformer", "norm_type": "layernorm"}


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
    x = model.forward_features(x)  #     type: ignore[operator]
    x = _drop_prefix_tokens(model, x)
    x = _to_nchw(x)
    return x


def _forward_get_intermediate_layers(model: Module, x: Tensor) -> Tensor:
    intermediates: Tensor = model.get_intermediate_layers(  # type: ignore[operator]
        x,
        n=1,  # Only return the n=1 last layers.
        reshape=True,  # Reshape the output to NCHW format.
        norm=True,  # Apply normalization to be consistent with forward_features.
    )
    return intermediates[0]


def _forward_intermediates(model: Module, x: Tensor) -> Tensor:
    intermediates: Tensor = model.forward_intermediates(  # type: ignore[operator]
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
