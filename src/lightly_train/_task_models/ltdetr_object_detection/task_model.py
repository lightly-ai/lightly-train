#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal, Union, cast

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.export import Dim
from typing_extensions import Self, override

from lightly_train._export import tensorrt_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer as DINOv2VisionTransformer,
)
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer as DINOv3VisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.ecvit.ecvit import ECViTModelWrapper
from lightly_train._models.ecvit.ecvit_package import EDGE_CRAFTER_PACKAGE
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionMetadata,
    ObjectDetectionOutput,
    ObjectDetectionPostprocessor,
    ObjectDetectionPreprocessor,
)
from lightly_train._task_models.ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
    DetectorConfig,
    DFINETransformerConfig,
    LTDETRDFINETransformerConfig,
    LTDETRRTDETRTransformerv2Config,
    RTDETRTransformerv2Config,
)
from lightly_train._task_models.ltdetr_object_detection.dino_vit_wrapper import (
    DINOSTAs,
)
from lightly_train._task_models.ltdetr_object_detection.dinov3_convnext_wrapper import (
    DINOv3ConvNextWrapper,
)
from lightly_train._task_models.ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)
from lightly_train._task_models.task_model import (
    ExportMixin,
    ONNXExportPrecisionPolicy,
    TaskModel,
)
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

LTDETR_DEFAULT_IMAGE_NORMALIZE: dict[str, tuple[float, ...]] = {
    "mean": (0.0, 0.0, 0.0),
    "std": (1.0, 1.0, 1.0),
}

_LTDETRDecoderName = Literal["rtdetrv2", "dfine"]
_TransformerConfig = Union[RTDETRTransformerv2Config, DFINETransformerConfig]
_TransformerConfigFactory = Callable[[], _TransformerConfig]


def _resolve_transformer_config(
    config: DetectorConfig, decoder_name: _LTDETRDecoderName | None
) -> _TransformerConfig:
    """Make backwards-compatible transformer config resolution for LTDETR task models."""
    resolved_decoder_name = decoder_name or config.transformer.decoder_name
    if resolved_decoder_name == config.transformer.decoder_name:
        return config.transformer

    config_name = type(config.transformer).__name__
    if resolved_decoder_name == "rtdetrv2":
        config_factory = cast(
            _TransformerConfigFactory,
            getattr(LTDETRRTDETRTransformerv2Config, config_name),
        )
    elif resolved_decoder_name == "dfine":
        config_factory = cast(
            _TransformerConfigFactory,
            getattr(LTDETRDFINETransformerConfig, config_name),
        )
    else:
        raise ValueError(
            f"Unsupported decoder_name={decoder_name!r}. "
            "Expected one of 'rtdetrv2' or 'dfine'."
        )
    return config_factory()


class LTDETRObjectDetection(TaskModel, ExportMixin):
    model_suffix = "ltdetr"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        patch_size: int | None = None,
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_freeze: bool = False,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        decoder_name: _LTDETRDecoderName | None = None,
        load_weights: bool = True,
    ) -> None:
        """Create an LTDETR object detection task model.

        Args:
            model_name:
                The model name. For example ``"dinov3/vits16-ltdetr"``.
            classes:
                A dict mapping class IDs to class names.
            image_size:
                The input image size.
            patch_size:
                Override for the backbone patch size used in ``resolve_auto``
                (stride computation). If None, the value from the model config's
                ``backbone_args`` is used.
            image_normalize:
                A dict containing normalization statistics with the keys ``"mean"``
                and ``"std"``.
            backbone_freeze:
                Whether to freeze the backbone during training.
            backbone_weights:
                Path to the backbone weights.
            backbone_args:
                Additional arguments merged into the backbone model args (override
                config defaults).
            decoder_name:
                Override the decoder from the model config.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        # Store init_args for checkpointing.
        super().__init__(init_args=locals(), ignore_args={"load_weights"})

        config: DetectorConfig = LTDETR_MODEL_REGISTRY.get(alias=model_name)()
        transformer_config = _resolve_transformer_config(
            config=config, decoder_name=decoder_name
        )
        config.transformer = transformer_config

        package_name, short_backbone = package_helpers.parse_model_name(
            config.backbone_name
        )
        self.image_size = image_size
        self.classes = classes
        self.backbone_freeze = backbone_freeze

        if backbone_freeze:
            config.backbone_wrapper.finetune = False

        # Use the config's baked-in patch_size unless the caller overrides it.
        if patch_size is None:
            patch_size = config.backbone_args.get("patch_size")
        else:
            config.backbone_args["patch_size"] = patch_size
        config.resolve_auto(patch_size=patch_size)

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This tensor maps the internal class id to the class id in `classes`. It is
        # owned by the postprocessor (constructed below).
        internal_class_to_class = torch.tensor(
            list(self.classes.keys()), dtype=torch.long
        )
        self.included_classes: dict[int, str] = {
            internal_class_id: class_name
            for internal_class_id, class_name in enumerate(self.classes.values())
        }

        self.image_normalize = (
            image_normalize
            if image_normalize is not None
            else dict(LTDETR_DEFAULT_IMAGE_NORMALIZE)
        )

        # Resolve the backbone's expected input channel count.
        # backbone_args["in_chans"] overrides image_normalize, which overrides 3.
        self._expected_input_channels: int
        if package_name == EDGE_CRAFTER_PACKAGE.name:
            self._expected_input_channels = 3
        elif backbone_args is not None and "in_chans" in backbone_args:
            self._expected_input_channels = backbone_args["in_chans"]
        else:
            self._expected_input_channels = len(self.image_normalize["mean"])

        # Build backbone model args: start from config defaults, then apply overrides.
        backbone_model_args: dict[str, Any] = dict(config.backbone_args)
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)
        is_dinov2_backbone = package_name == DINOV2_VIT_PACKAGE.name
        load_dinov2_backbone_weights = (
            load_weights and backbone_weights is not None and is_dinov2_backbone
        )
        if backbone_weights is not None and not is_dinov2_backbone:
            backbone_model_args["weights"] = str(backbone_weights)

        get_model_kwargs = {
            "num_input_channels": len(self.image_normalize["mean"]),
        }

        package = package_helpers.get_package(package_name)

        backbone = package.get_model(
            model_name=short_backbone,
            model_args=backbone_model_args,
            load_weights=load_weights,
            **get_model_kwargs,
        )
        assert isinstance(
            backbone,
            (
                ConvNeXt,
                DINOv3VisionTransformer,
                DINOv2VisionTransformer,
                ECViTModelWrapper,
            ),
        )
        if load_dinov2_backbone_weights:
            assert backbone_weights is not None
            self.load_backbone_weights(backbone=backbone, path=backbone_weights)

        self.backbone: DINOSTAs | DINOv3ConvNextWrapper | ECViTBackboneWrapper

        if isinstance(backbone, ECViTModelWrapper):
            self.backbone = ECViTBackboneWrapper(model_wrapper=backbone)
        elif isinstance(backbone, (DINOv3VisionTransformer, DINOv2VisionTransformer)):
            # TODO(Guarin, 02/26): Improve how mask tokens are handled for fine-tuning.
            backbone.mask_token.requires_grad = False  # type: ignore
            vit_model_wrapper: DINOv2ViTModelWrapper | DINOv3ViTModelWrapper = (
                DINOv2ViTModelWrapper(backbone)
                if isinstance(backbone, DINOv2VisionTransformer)
                else DINOv3ViTModelWrapper(backbone)
            )
            self.backbone = DINOSTAs(
                model_wrapper=vit_model_wrapper,
                **config.backbone_wrapper.model_dump(exclude={"conv_inplane_factor"}),
            )
        else:
            assert isinstance(backbone, ConvNeXt)
            convnext_model_wrapper = DINOv3VConvNeXtModelWrapper(backbone)
            self.backbone = DINOv3ConvNextWrapper(model_wrapper=convnext_model_wrapper)

        self.encoder: HybridEncoder = HybridEncoder(
            **config.hybrid_encoder.model_dump()
        )

        transformer_cfg = transformer_config.model_dump(exclude={"decoder_name"})
        transformer_cfg["num_classes"] = len(self.classes)
        if transformer_config.decoder_name == "rtdetrv2":
            self.decoder: RTDETRTransformerv2 | DFINETransformer = RTDETRTransformerv2(  # type: ignore[no-untyped-call]
                **transformer_cfg, eval_spatial_size=self.image_size
            )
        else:
            self.decoder = DFINETransformer(  # type: ignore[no-untyped-call]
                **transformer_cfg, eval_spatial_size=self.image_size
            )

        self.preprocessor = ObjectDetectionPreprocessor(
            image_size=self.image_size,
            image_normalize=self.image_normalize,
            expected_input_channels=self._expected_input_channels,
        )
        self.postprocessor = ObjectDetectionPostprocessor(
            num_classes=len(self.classes),
            num_top_queries=config.rtdetr_postprocessor.num_top_queries,
            internal_class_to_class=internal_class_to_class,
        )

        # Tracks whether `deploy()` (in-place reparameterization for faster
        # inference) has been applied. `predict*` reparameterizes lazily on first
        # use; correctness does not depend on it.
        self._deployed = False

        if self.backbone_freeze:
            self.freeze_backbone()

    @override
    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        return model in LTDETR_MODEL_REGISTRY.list_aliases()

    @classmethod
    def list_model_names(cls) -> list[str]:
        return list(LTDETR_MODEL_REGISTRY.list_aliases())

    @classmethod
    def parse_model_name(cls, model_name: str) -> dict[str, str]:
        """Resolve a registered model alias into its package/backbone parts."""
        try:
            config = LTDETR_MODEL_REGISTRY.get(alias=model_name)()
        except KeyError:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available "
                f"models are: {cls.list_model_names()}."
            )
        package_name, backbone_name = package_helpers.parse_model_name(
            config.backbone_name
        )
        return {
            "package_name": package_name,
            "model_name": f"{package_name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
        }

    def get_export_output_names(self) -> list[str]:
        return ["logits", "boxes"]

    @property
    @override
    def model_input_spec(self) -> ModelInputSpec:
        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(
                        self._expected_input_channels,
                        self.image_size[0],
                        self.image_size[1],
                    ),
                    dtype=torch.float32,
                    is_batched=True,
                )
            },
            input_dynamic_shapes={
                "images": (
                    Dim("batch_size", min=1, max=2**31 - 1),
                    Dim.STATIC,
                    Dim.STATIC,
                    Dim.STATIC,
                )
            },
        )

    @property
    @override
    def onnx_export_precision_policy(self) -> ONNXExportPrecisionPolicy:
        # Keep modules called from disabled-autocast decoder regions in FP32 for
        # FP16 export. Keep attention-adjacent ONNX ops in FP32 as large Softmax
        # inputs can otherwise overflow to NaN, and blocking MatMul preserves the
        # Cast nodes that keep TensorRT strongly typed.
        fp32_module_names = ["decoder.dec_bbox_head"]
        if isinstance(self.decoder, DFINETransformer):
            fp32_module_names.extend(
                [
                    "decoder.pre_bbox_head",
                    "decoder.dec_score_head",
                    "decoder.decoder.lqe_layers",
                ]
            )
        return ONNXExportPrecisionPolicy(
            fp32_module_names=tuple(fp32_module_names),
            fp32_onnx_op_types=("Softmax", "MatMul"),
        )

    @override
    def verify_onnx_export_outputs(
        self,
        *,
        torch_outputs: BaseModelOutput,
        onnx_outputs: BaseModelOutput,
    ) -> None:
        if not isinstance(torch_outputs, ObjectDetectionOutput) or not isinstance(
            onnx_outputs, ObjectDetectionOutput
        ):
            raise TypeError(
                "LTDETRObjectDetection ONNX verification expects "
                "ObjectDetectionOutput instances."
            )

        for output_name in ("logits", "boxes"):
            output_model = getattr(torch_outputs, output_name)
            output_onnx = getattr(onnx_outputs, output_name)

            def msg(s: str) -> str:
                return f'ONNX validation failed for output "{output_name}": {s}'

            # Query order can differ across backends while preserving equivalent
            # predictions, so compare the query-reduced tensors.
            output_model = output_model.sum(dim=1)
            if output_onnx.is_floating_point():
                output_onnx = output_onnx.float()
            output_onnx = output_onnx.sum(dim=1)

            torch.testing.assert_close(
                output_onnx,
                output_model,
                msg=msg,
                equal_nan=True,
                check_device=False,
                check_dtype=False,
                check_layout=False,
                atol=5e-3,
                rtol=1e-1,
            )

    @override
    def _onnx_export_input_spec(
        self,
        *,
        spec: ModelInputSpec,
        height: int | None,
        width: int | None,
    ) -> ModelInputSpec:
        export_height = self.image_size[0] if height is None else height
        export_width = self.image_size[1] if width is None else width
        if (export_height, export_width) != self.image_size:
            raise ValueError(
                "LTDETRObjectDetection ONNX export does not support custom "
                "height/width values because decoder anchors are tied to image_size. "
                f"Expected height={self.image_size[0]} and width={self.image_size[1]}, "
                f"got height={export_height} and width={export_width}."
            )
        return super()._onnx_export_input_spec(spec=spec, height=height, width=width)

    def forward_backend(self, images: Tensor) -> ObjectDetectionOutput:
        # For backwards compatibility with the benchmark command.
        out: ObjectDetectionOutput = self(images)
        return out

    def forward(self, images: Tensor) -> ObjectDetectionOutput:
        # The raw neural forward pass. Returns the raw decoder outputs:
        #   logits: (B, num_queries, num_classes)
        #   boxes:  (B, num_queries, 4) in normalized cxcywh format
        # Top-k selection, thresholding, NMS and rescaling to original image
        # coordinates are left to the caller (see `postprocess`/`predict*`).
        x = self.backbone(images)
        x = self.encoder(x)
        out = self.decoder(x)
        return ObjectDetectionOutput(logits=out["pred_logits"], boxes=out["pred_boxes"])

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    @property
    @override
    def is_deploy_mode(self) -> bool:
        return self._deployed

    @override
    def deploy(self) -> Self:
        # Reparameterizes modules in place (RepVGG conv fusion, D-FINE layer
        # pruning) for faster inference. This is output-equivalent to the eval-mode
        # forward pass, so it is an optimization rather than a correctness
        # requirement.
        self.eval()
        if self._deployed:
            # convert_to_deploy() is NOT idempotent (D-FINE re-slices its layers), so
            # guard the reparameterization. eval() above still runs every call.
            return self
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        self._deployed = True
        return self

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load the state dict from a training checkpoint.

        Loads the EMA weights if available, otherwise falls back to the model weights.
        """
        has_ema_weights = any(k.startswith("ema_model.model.") for k in state_dict)
        has_model_weights = any(k.startswith("model.") for k in state_dict)
        new_state_dict = {}
        if has_ema_weights:
            for name, param in state_dict.items():
                if name.startswith("ema_model.model."):
                    name = name[len("ema_model.model.") :]
                    new_state_dict[name] = param
        elif has_model_weights:
            for name, param in state_dict.items():
                if name.startswith("model."):
                    name = name[len("model.") :]
                    new_state_dict[name] = param
        return self.load_state_dict(new_state_dict, strict=strict, assign=assign)

    def load_backbone_weights(self, backbone: torch.nn.Module, path: PathLike) -> None:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights file not found: '{path}'")

        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(f"Missing keys when loading backbone: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading backbone: {unexpected}")
        if not missing and not unexpected:
            logger.info(f"Backbone weights loaded from '{path}'")

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        threshold: float = 0.6,
    ) -> list[dict[str, Tensor]]:
        """Run inference on a batch of images and return per-image predictions.

        Args:
            images:
                Sequence of input images. Each can be a path, a PIL image, or a
                tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions
                with scores <= threshold are discarded.

        Returns:
            A list with one prediction dict per input image.
        """
        self._track_inference()
        if self.training or not self.is_deploy_mode:
            self.deploy()
        first_param = next(self.parameters())
        tensors: list[Tensor] = []
        metadata: list[ObjectDetectionMetadata] = []
        for image in images:
            x, meta = self.preprocessor.preprocess_image(
                image, device=first_param.device, dtype=first_param.dtype
            )
            tensors.append(x)
            metadata.append(meta)
        batch = torch.stack(tensors, dim=0)
        batch = self.preprocessor.preprocess_batch(batch)
        raw = self(batch)
        return self.postprocessor.postprocess(raw, metadata, threshold=threshold)

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.6
    ) -> dict[str, Tensor]:
        """Run inference on a single image and return task-specific predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions with
                scores <= threshold are discarded.

        Returns:
            A task-specific prediction dictionary.
        """
        self._track_inference()
        if self.training or not self.is_deploy_mode:
            self.deploy()
        first_param = next(self.parameters())
        x, metadata = self.preprocessor.preprocess_image(
            image, device=first_param.device, dtype=first_param.dtype
        )
        batch = self.preprocessor.preprocess_batch(x.unsqueeze(0))
        raw = self(batch)
        return self.postprocessor.postprocess(raw, [metadata], threshold=threshold)[0]

    @torch.no_grad()
    def predict_sahi(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.6,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
    ) -> dict[str, Tensor]:
        """Run Slicing Aided Hyper Inference (SAHI) inference on the input image.

        The image is first converted to a tensor, then:

        - Tiled into overlapping crops of size `self.image_size`.
        - A resized full-image version is added as a "global" tile.
        - All tiles (global + local) are passed through the model in parallel.
        - Predictions are filtered by score and merged using NMS and a
          global/local consistency heuristic. NMS is only applied on tiles predictions.
          The heuristic discards tiles predictions that heavily overlaps with global
          predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold for filtering low-confidence predictions.
            overlap:
                Fractional overlap between tiles in [0, 1). 0.0 means no overlap.
            nms_iou_threshold:
                IoU threshold used for non-maximum suppression when merging
                predictions from tiles and global image. A lower nms_iou_threshold
                value yields less predictions.
            global_local_iou_threshold:
                Minimum IoU required to consider a tile prediction
                as matching a global prediction when combining them. A lower
                global_local_iou_threshold yields less predictions.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max)
                  in the coordinates of the original image.
                - "scores": Tensor of shape (N,) with confidence scores for each prediction.
        """

        if self.training or not self.is_deploy_mode:
            self.deploy()

        first_param = next(self.parameters())
        batch, metadata = self.preprocessor.preprocess_sahi_image(
            image,
            device=first_param.device,
            dtype=first_param.dtype,
            overlap=overlap,
        )
        raw = self(batch)
        return self.postprocessor.postprocess_sahi(
            raw,
            metadata,
            threshold=threshold,
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
            tile_size=self.image_size,
        )

    def export_tensorrt(
        self,
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        onnx_args: dict[str, Any] | None = None,
        max_batchsize: int = 1,
        opt_batchsize: int = 1,
        min_batchsize: int = 1,
        verbose: bool = False,
    ) -> None:
        """Build a TensorRT engine from an ONNX model.

        .. note::
            TensorRT is not part of LightlyTrain’s dependencies and must be installed separately.
            Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup.
            See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details.
            On CUDA 12.x systems you can often install the Python package via `pip install tensorrt-cu12`.

        This loads the ONNX file, parses it with TensorRT, infers the static input
        shape (C, H, W) from the `"images"` input, and creates an engine with a
        dynamic batch dimension in the range `[min_batchsize, opt_batchsize, max_batchsize]`.
        Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

        The engine is serialized and written to `out`.

        Args:
            out:
                Path where the TensorRT engine will be saved.
            precision:
                Precision for ONNX export and TensorRT engine building. Either
                "fp32" or "fp16".
            onnx_args:
                Optional arguments to pass to `export_onnx` when exporting
                the ONNX model prior to building the TensorRT engine. If None,
                default arguments are used and the ONNX file is saved alongside
                the TensorRT engine with the same name but `.onnx` extension.
            max_batchsize:
                Maximum supported batch size.
            opt_batchsize:
                Batch size TensorRT optimizes for.
            min_batchsize:
                Minimum supported batch size.
            verbose:
                Enable verbose TensorRT logging.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
            RuntimeError: If the ONNX cannot be parsed or engine building fails.
            ValueError: If batch size constraints are invalid or H/W are dynamic.
        """
        model_dtype = next(self.parameters()).dtype

        onnx_args = dict(onnx_args) if onnx_args is not None else {}
        onnx_args.setdefault("precision", precision)

        tensorrt_helpers.export_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=model_dtype,
            onnx_args=onnx_args,
            max_batchsize=max_batchsize,
            opt_batchsize=opt_batchsize,
            min_batchsize=min_batchsize,
            # We convert the fp32 attention scores already during ONNX export, so we
            # build a strongly-typed engine: TensorRT then honors those fp32 Cast nodes
            # instead of forcing the whole attention into FP16 (which overflows to NaN).
            fp32_attention_scores=False,
            strongly_typed=True,
            verbose=verbose,
        )
