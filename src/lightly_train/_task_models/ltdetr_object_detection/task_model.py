#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Literal, Union, cast

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from typing_extensions import Self, override

from lightly_train import _logging
from lightly_train._commands import _warnings
from lightly_train._export import tensorrt_helpers
from lightly_train._export.onnx_helpers import (
    check_onnx_dynamo_requirements,
    fix_topological_order,
    remove_redundant_casts,
    write_onnx_metadata,
)
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer as DINOv2VisionTransformer,
)
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
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
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import ModelInputSpec
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


class LTDETRObjectDetection(TaskModel):
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
        self._config = config
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
        self._backbone_package_name = package_name
        self._backbone_name = short_backbone

        if backbone_freeze:
            config.backbone_wrapper.finetune = False

        # Use the config's baked-in patch_size unless the caller overrides it.
        if patch_size is None:
            patch_size = config.backbone_args.get("patch_size")
        else:
            config.backbone_args["patch_size"] = patch_size
        config.resolve_auto(patch_size=patch_size)

        # Internally, the model processes classes as contiguous integers starting at 0.
        # The postprocessor owns the mapping back to user-facing class ids.
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
        self._deployed = False

        if self.backbone_freeze:
            self.freeze_backbone()

    @override
    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        return model in LTDETR_MODEL_REGISTRY.list_aliases()

    @classmethod
    def list_model_names(cls) -> list[str]:
        return LTDETR_MODEL_REGISTRY.list_model_names()

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

    @property
    def model_input_spec(self) -> ModelInputSpec:
        return self._config.model_input_spec(
            image_size=self.image_size,
            input_channels=self._expected_input_channels,
        )

    def get_export_output_names(self) -> list[str]:
        return ["logits", "boxes"]

    def forward_backend(self, x: Tensor) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)

    def forward(self, x: Tensor) -> ObjectDetectionOutput:
        raw = self.forward_backend(x)
        return ObjectDetectionOutput(logits=raw["pred_logits"], boxes=raw["pred_boxes"])

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: Any | dict[str, Tensor],
        metadata: Sequence[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Tensor]]:
        if isinstance(raw_outputs, dict):
            raw = ObjectDetectionOutput(
                logits=raw_outputs["pred_logits"], boxes=raw_outputs["pred_boxes"]
            )
        else:
            raw = cast(ObjectDetectionOutput, raw_outputs)
        typed_metadata = cast(Sequence[ObjectDetectionMetadata], metadata)
        return self.postprocessor.postprocess(raw, typed_metadata, threshold)

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def deploy(self) -> Self:
        self.eval()
        if self._deployed:
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
        if self.training or not self._deployed:
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
        batch = self.preprocessor.preprocess_batch(torch.stack(tensors, dim=0))
        return self.postprocessor.postprocess(self(batch), metadata, threshold)

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
        if self.training or not self._deployed:
            self.deploy()
        first_param = next(self.parameters())
        x, metadata = self.preprocessor.preprocess_image(
            image, device=first_param.device, dtype=first_param.dtype
        )
        batch = self.preprocessor.preprocess_batch(x.unsqueeze(0))
        return self.postprocessor.postprocess(self(batch), [metadata], threshold)[0]

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

        if self.training or not self._deployed:
            self.deploy()
        first_param = next(self.parameters())
        batch, metadata = self.preprocessor.preprocess_sahi_image(
            image,
            device=first_param.device,
            dtype=first_param.dtype,
            overlap=overlap,
        )
        raw = self(self.preprocessor.preprocess_sahi_batch(batch))
        return self.postprocessor.postprocess_sahi(
            raw,
            metadata,
            threshold=threshold,
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
            tile_size=self.image_size,
        )

    @torch.no_grad()
    def predict_sahi_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        threshold: float = 0.6,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
    ) -> list[dict[str, Tensor]]:
        """Run Slicing Aided Hyper Inference on a batch of images."""
        self._track_inference()
        if not images:
            raise ValueError("images must contain at least one image.")
        if self.training or not self._deployed:
            self.deploy()
        first_param = next(self.parameters())
        batches: list[Tensor] = []
        metadata: list[ObjectDetectionMetadata] = []
        batch_sizes: list[int] = []
        for image in images:
            image_batch, image_metadata = self.preprocessor.preprocess_sahi_image(
                image,
                device=first_param.device,
                dtype=first_param.dtype,
                overlap=overlap,
            )
            batches.append(image_batch)
            metadata.append(image_metadata)
            batch_sizes.append(len(image_batch))
        raw = self(self.preprocessor.preprocess_sahi_batch(torch.cat(batches, dim=0)))
        out: list[dict[str, Tensor]] = []
        start = 0
        for image_metadata, batch_size in zip(metadata, batch_sizes):
            end = start + batch_size
            raw_image = ObjectDetectionOutput(
                logits=raw.logits[start:end], boxes=raw.boxes[start:end]
            )
            out.append(
                self.postprocessor.postprocess_sahi(
                    raw_image,
                    image_metadata,
                    threshold=threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    global_local_iou_threshold=global_local_iou_threshold,
                    tile_size=self.image_size,
                )
            )
            start = end
        return out

    def onnx_export_metadata(self) -> dict[str, str]:
        """Return metadata embedded in exported LT-DETR ONNX models."""
        from lightly_train import __version__
        from lightly_train._license import LICENSE_INFO

        metadata = {
            "lightly_train_version": __version__,
            "license_info": LICENSE_INFO,
        }
        if self.image_normalize is not None:
            metadata["image_normalize"] = json.dumps(
                self.image_normalize, sort_keys=True
            )
        metadata["classes"] = json.dumps(self.classes, sort_keys=True)
        model_name = self.init_args.get("model_name")
        if model_name is not None:
            metadata["model_name"] = str(model_name)
        return metadata

    @torch.no_grad()
    def export_onnx(
        self,
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
        num_channels: int | None = None,
    ) -> None:
        """Export the model to ONNX using its declared model I/O specification."""
        _warnings.filter_export_warnings()
        _logging.set_up_console_logging()
        check_onnx_dynamo_requirements()

        if precision not in ("fp32", "fp16"):
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'fp32', 'fp16'."
            )
        if precision == "fp16" and not simplify:
            raise ValueError("fp16 precision requires simplify=True.")

        # Trace in fp32. The decoder contains disabled-autocast regions, and the
        # existing graph conversion below selectively preserves sensitive ops in fp32.
        self.eval().to(torch.float32)
        self.deploy()
        model_device = next(self.parameters()).device

        spec = self.model_input_spec
        trace_batch_size = 2 if dynamic_batch_size else batch_size
        example_inputs = spec.example_inputs(
            batch_size=trace_batch_size,
            device=model_device,
            dtype=torch.float32,
        )
        if num_channels is not None:
            images = example_inputs["images"]
            example_inputs["images"] = torch.randn(
                trace_batch_size,
                num_channels,
                *images.shape[-2:],
                device=model_device,
                dtype=torch.float32,
            )
        dynamic_shapes = spec.dynamic_shapes(dynamic_batch_size=dynamic_batch_size)

        with torch.no_grad():
            program = torch.export.export(
                self,
                args=(),
                kwargs=example_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
            )
            example_output = self(**example_inputs)

        input_names = list(spec.input_specs)
        output_names = [field.name for field in fields(example_output)]
        logger.info(f"Exporting ONNX model to '{out}'")
        torch.onnx.export(
            program,
            f=str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            **(format_args or {}),
        )

        if precision == "fp16":
            import onnx
            from onnxruntime.transformers import float16 as ort_float16

            model_onnx = onnx.load(str(out))
            op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + [
                "Softmax",
                "MatMul",
            ]
            model_fp16 = ort_float16.convert_float_to_float16(
                model_onnx, op_block_list=op_block_list
            )
            remove_redundant_casts(model_fp16)
            fix_topological_order(model_fp16)
            onnx.save(model_fp16, str(out))

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                str(out),
                output_model=out,
                skip_optimizations=["constant_folding"],
            )

        # Graph conversion and simplification can drop metadata, so write it last.
        write_onnx_metadata(out=out, metadata=self.onnx_export_metadata())

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(str(out), full_check=True)
            providers = ort.get_available_providers()
            if precision == "fp16" and "CUDAExecutionProvider" not in providers:
                logger.warning(
                    "Skipping ONNX runtime verification for fp16 model because "
                    "CUDAExecutionProvider is not available in onnxruntime. "
                    "Install onnxruntime-gpu to enable full verification."
                )
            else:
                reference_model = deepcopy(self).cpu().to(torch.float32).eval()
                reference_model.deploy()
                reference_inputs = {
                    name: tensor.detach().cpu().to(torch.float32)
                    for name, tensor in example_inputs.items()
                }
                reference_output = reference_model(**reference_inputs)
                reference_values = [
                    getattr(reference_output, field.name)
                    for field in fields(reference_output)
                ]

                session = ort.InferenceSession(str(out))
                session_input_types = {
                    input_.name: input_.type for input_ in session.get_inputs()
                }
                input_feed = {}
                for name, tensor in example_inputs.items():
                    tensor = tensor.detach().cpu()
                    if session_input_types.get(name) == "tensor(float16)":
                        tensor = tensor.half()
                    input_feed[name] = tensor.numpy()
                outputs_onnx = [
                    torch.from_numpy(output) for output in session.run(None, input_feed)
                ]

                if len(outputs_onnx) != len(reference_values):
                    raise AssertionError(
                        "Number of ONNX outputs should be "
                        f"{len(reference_values)} but is {len(outputs_onnx)}"
                    )
                for output_onnx, output_model, output_name in zip(
                    outputs_onnx, reference_values, output_names
                ):

                    def msg(message: str) -> str:
                        return (
                            f'ONNX validation failed for output "{output_name}": '
                            f"{message}"
                        )

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

        logger.info(f"Successfully exported ONNX model to '{out}'")

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

        strongly_typed = (
            precision == "fp16"
            and self._backbone_package_name == DINOV3_PACKAGE.name
            and self._backbone_name == "vits16"
        )
        if strongly_typed:
            logger.info(
                "Using a strongly-typed TensorRT network for DINOv3 ViT-S to "
                "preserve FP32 attention scores and prevent FP16 overflow."
            )
        else:
            logger.info(
                "Using a weakly-typed TensorRT network so TensorRT can optimize "
                "layer precision."
            )

        tensorrt_helpers.export_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=model_dtype,
            onnx_args=onnx_args,
            max_batchsize=max_batchsize,
            opt_batchsize=opt_batchsize,
            min_batchsize=min_batchsize,
            fp32_attention_scores=False,
            strongly_typed=strongly_typed,
            verbose=verbose,
        )
