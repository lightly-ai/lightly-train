#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.ops import box_convert
from typing_extensions import Self

from lightly_train._export import tensorrt_helpers
from lightly_train._export.export_onnx import (
    ONNXExportMixin,
    ONNXExportPrecisionPolicy,
)
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    ObjectDetectionMetadata,
    ObjectDetectionPostprocessor,
    ObjectDetectionPrediction,
    ObjectDetectionPreprocessor,
    ObjectDetectionSAHIConfig,
)
from lightly_train._task_models.picodet_object_detection.config import (
    PICODET_OBJECT_DETECTION_MODEL_REGISTRY,
)
from lightly_train._task_models.picodet_object_detection.csp_pan import CSPPAN
from lightly_train._task_models.picodet_object_detection.esnet import ESNet
from lightly_train._task_models.picodet_object_detection.pico_head import (
    PicoHead,
    distance2bbox,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.task_model_io import BaseModelOutput, ModelInputSpec
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class PicoDetObjectDetection(TaskModel, ONNXExportMixin):
    """PicoDet-S object detection model.

    PicoDet is a lightweight anchor-free object detector designed for
    mobile and edge deployment. It uses an Enhanced ShuffleNet backbone,
    CSP-PAN neck, and GFL-style detection head.
    """

    model_suffix = "picodet"

    def __init__(
        self,
        *,
        model_name: str,
        image_size: tuple[int, int],
        num_classes: int,
        classes: dict[int, str] | None = None,
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        reg_max: int = 7,
        score_threshold: float = 0.025,
        iou_threshold: float = 0.6,
        max_detections: int = 100,
        backbone_weights: PathLike | None = None,
        load_weights: bool = True,
        backbone_freeze: bool = False,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"backbone_weights", "load_weights"}
        )

        self.model_name = model_name
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.classes = classes
        self.backbone_freeze = backbone_freeze
        # Kept as attributes so the train model can build its dense-head decoder from
        # the same values, without them having to be duplicated in the train args.
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        if classes is not None and len(classes) != num_classes:
            raise ValueError("classes must have the same length as num_classes.")

        internal_class_to_class = (
            list(range(num_classes)) if classes is None else list(classes.keys())
        )
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,
        )
        self.included_classes: dict[int, str] = (
            {i: str(i) for i in range(num_classes)}
            if classes is None
            else {
                internal_class_id: class_name
                for internal_class_id, class_name in enumerate(classes.values())
            }
        )

        try:
            config = PICODET_OBJECT_DETECTION_MODEL_REGISTRY.get(alias=model_name)()
        except KeyError as error:
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                f"Available: {list(PICODET_OBJECT_DETECTION_MODEL_REGISTRY.list_aliases())}"
            ) from error
        self._config = config

        self.backbone = ESNet(
            model_size=config.model_size,
            out_indices=(2, 9, 12),  # C3, C4, C5
        )
        backbone_out_channels = self.backbone.out_channels

        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

        if self.backbone_freeze:
            self.freeze_backbone()

        self.neck = CSPPAN(
            in_channels=backbone_out_channels,
            out_channels=config.neck_out_channels,
            kernel_size=5,
            num_features=4,  # P3, P4, P5, P6
            expansion=1.0,
            num_csp_blocks=1,
            use_depthwise=True,
        )

        self.head = PicoHead(
            in_channels=config.neck_out_channels,
            num_classes=num_classes,
            feat_channels=config.head_feat_channels,
            stacked_convs=config.stacked_convs,
            kernel_size=5,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            share_cls_reg=True,
            use_depthwise=True,
        )
        self.o2o_head = PicoHead(
            in_channels=config.neck_out_channels,
            num_classes=num_classes,
            feat_channels=config.head_feat_channels,
            stacked_convs=config.stacked_convs,
            kernel_size=5,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            share_cls_reg=True,
            use_depthwise=True,
        )

        # The decoder takes a flat top-k over (anchor, class) pairs, so the cap cannot
        # exceed the number of available pairs. Only reachable at tiny image sizes.
        num_anchors = sum(
            math.ceil(image_size[0] / stride) * math.ceil(image_size[1] / stride)
            for stride in (8, 16, 32, 64)
        )
        self.num_top_queries = min(max_detections, num_anchors * num_classes)

        # Grayscale inputs are expanded to this many channels by the preprocessor.
        self._expected_input_channels = (
            3 if image_normalize is None else len(image_normalize["mean"])
        )
        self.preprocessor = ObjectDetectionPreprocessor(
            image_size=image_size,
            image_normalize=image_normalize,
            expected_input_channels=self._expected_input_channels,
        )
        self.postprocessor = ObjectDetectionPostprocessor(
            num_classes=num_classes,
            num_top_queries=self.num_top_queries,
            internal_class_to_class=self.internal_class_to_class,
            image_size=image_size,
        )
        self._deployed = False

        self._o2o_peak_score_thresholds = (0.005, 0.02, 0.04, 0.06)
        self._o2o_peak_kernels = (3, 3, 5, 5)
        self._o2o_suppress_logit = -1e6

    def _apply_o2o_peak_filter(self, cls_score: Tensor, level_idx: int) -> Tensor:
        """Suppress non-peak logits to sparsify dense o2o predictions."""
        scores = cls_score.sigmoid().amax(dim=1, keepdim=True)
        threshold = self._o2o_peak_score_thresholds[level_idx]
        kernel = self._o2o_peak_kernels[level_idx]
        pooled = F.max_pool2d(scores, kernel_size=kernel, stride=1, padding=kernel // 2)
        keep = (scores >= threshold) & (scores == pooled)
        suppressed = cls_score.new_full((), self._o2o_suppress_logit)
        return torch.where(keep, cls_score, suppressed)

    @property
    def model_input_spec(self) -> ModelInputSpec:
        return self._config.model_input_spec(
            image_size=self.image_size,
            input_channels=self._expected_input_channels,
        )

    @property
    def onnx_export_precision_policy(self) -> ONNXExportPrecisionPolicy:
        # The DFL "Integral" expectation (a Softmax over the reg_max + 1 bins followed
        # by a projection onto the bin centers) is the only part of the decode whose
        # precision directly moves box coordinates, so keep it in FP32. The grid
        # centers fold into constants that are exact in FP16 because they are
        # multiples of the stride, and the remaining decode is additions on values
        # bounded by the input size. Torch exports the projection as Gemm here;
        # MatMul is listed defensively in case a later opset emits that instead.
        return ONNXExportPrecisionPolicy(
            fp32_onnx_op_types=("Softmax", "Gemm", "MatMul")
        )

    @property
    def is_deploy_mode(self) -> bool:
        return self._deployed

    def deploy(self) -> Self:
        self.eval()
        if self._deployed:
            return self
        # No PicoDet submodule implements convert_to_deploy today. The sweep is kept
        # so that re-parameterizable blocks added later are picked up automatically,
        # and because the ONNX export pipeline and the benchmark Torch backend rely
        # on deploy() putting the model into eval mode.
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        self._deployed = True
        return self

    def load_backbone_weights(self, path: PathLike) -> None:
        """Load backbone weights from a checkpoint file.

        Args:
            path: Path to a .pt file (e.g., exported_last.pt).
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights file not found: '{path}'")

        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "model_state_dict", "student"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

        if isinstance(state_dict, dict):
            if all(key.startswith("module.") for key in state_dict):
                state_dict = {
                    key[len("module.") :]: value for key, value in state_dict.items()
                }

            prefixes = ("_model.", "model.", "backbone.")
            if all(key.startswith(prefixes) for key in state_dict):
                state_dict = {
                    key.split(".", 1)[1]: value for key, value in state_dict.items()
                }
            elif any(key.startswith(prefixes) for key in state_dict):
                state_dict = {
                    key.split(".", 1)[1]: value
                    for key, value in state_dict.items()
                    if key.startswith(prefixes)
                }

        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        total_backbone_keys = len(self.backbone.state_dict())
        loaded_keys = total_backbone_keys - len(missing)
        logger.info(
            "Backbone weights loaded: %d/%d keys matched.",
            loaded_keys,
            total_backbone_keys,
        )
        if missing:
            logger.warning("Missing keys when loading backbone: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading backbone: %s", unexpected)

    @classmethod
    def list_model_names(cls) -> list[str]:
        """Return list of supported model names."""
        return list(PICODET_OBJECT_DETECTION_MODEL_REGISTRY.list_aliases())

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        """Check if a model name is supported."""
        return model in PICODET_OBJECT_DETECTION_MODEL_REGISTRY.list_aliases()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load the state dict from a training checkpoint.

        Loads EMA weights if available, otherwise falls back to model weights.

        Args:
            state_dict: Checkpoint state dict.
            strict: Whether to strictly enforce key matching.
            assign: Whether to assign parameters instead of copying.

        Returns:
            Incompatible keys from loading.
        """
        has_ema_weights = any(k.startswith("ema_model.model.") for k in state_dict)
        has_model_weights = any(k.startswith("model.") for k in state_dict)

        new_state_dict = {}
        if has_ema_weights:
            for name, param in state_dict.items():
                if name.startswith("ema_model.model."):
                    new_name = name[len("ema_model.model.") :]
                    new_state_dict[new_name] = param
        elif has_model_weights:
            for name, param in state_dict.items():
                if name.startswith("model."):
                    new_name = name[len("model.") :]
                    new_state_dict[new_name] = param
        else:
            new_state_dict = state_dict

        # internal_class_to_class is a non-persistent buffer initialized in
        # __init__, but training checkpoints may include it. Remove to avoid
        # unexpected key errors.
        new_state_dict.pop("internal_class_to_class", None)
        return self.load_state_dict(new_state_dict, strict=strict, assign=assign)

    def _forward_train(self, images: Tensor) -> dict[str, Tensor | list[Tensor]]:
        """Forward pass returning raw per-level predictions.

        Args:
            images: Input tensor of shape (B, C, H, W).

        Returns:
            Dictionary with:
            - cls_scores: List of (B, num_classes, H, W) per level.
            - bbox_preds: List of (B, 4*(reg_max+1), H, W) per level.
        """
        feats = self.backbone(images)
        feats = self.neck(feats)
        cls_scores, bbox_preds = self.head(feats)
        o2o_cls_scores, o2o_bbox_preds = self.o2o_head(feats)
        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
            "o2o_cls_scores": o2o_cls_scores,
            "o2o_bbox_preds": o2o_bbox_preds,
        }

    def decode_o2o_outputs(
        self,
        *,
        cls_scores_list: list[Tensor],
        bbox_preds_list: list[Tensor],
        input_size: tuple[int, int],
    ) -> ObjectDetectionBatchOutput:
        """Decode dense o2o head outputs into raw logits and normalized boxes.

        Args:
            cls_scores_list: Per-level ``(B, num_classes, H, W)`` class logits.
            bbox_preds_list: Per-level ``(B, 4*(reg_max+1), H, W)`` DFL logits.
            input_size: ``(height, width)`` of the model input the boxes refer to.

        Returns:
            An :class:`ObjectDetectionBatchOutput` with ``logits`` of shape ``(B, N, C)``
            (raw, pre-sigmoid) and ``boxes`` of shape ``(B, N, 4)`` in normalized
            ``cxcywh`` relative to the model input. ``N`` is the total number of
            anchor points over all stride levels.
        """
        batch_size = cls_scores_list[0].shape[0]
        device = cls_scores_list[0].device
        dtype = cls_scores_list[0].dtype
        decoded_boxes_pixel: list[Tensor] = []
        flat_cls_logits: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(
            zip(cls_scores_list, bbox_preds_list)
        ):
            stride = self.o2o_head.strides[level_idx]
            _, _, h, w = cls_score.shape
            num_points = h * w

            cls_score = self._apply_o2o_peak_filter(cls_score, level_idx)
            # Grid centers in model-input pixels. These fold into ONNX constants that
            # stay exact in FP16 because they are multiples of the stride.
            y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
            x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

            center_in_feature = points / stride
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, 4 * (self.reg_max + 1)
            )
            pred_corners = self.o2o_head.integral(bbox_pred_flat)
            decode_bbox_pred = distance2bbox(
                center_in_feature.unsqueeze(0).expand(batch_size, -1, -1), pred_corners
            )
            decoded_boxes_pixel.append(decode_bbox_pred * stride)

            cls_pred_flat = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, self.num_classes
            )
            flat_cls_logits.append(cls_pred_flat)

        boxes_xyxy = torch.cat(decoded_boxes_pixel, dim=1)
        logits = torch.cat(flat_cls_logits, dim=1)

        # Normalize to the model input before converting to cxcywh. The two operations
        # commute, but clamping to the image rectangle is only meaningful in xyxy.
        input_h, input_w = input_size
        scale = boxes_xyxy.new_tensor([input_w, input_h, input_w, input_h])
        boxes_xyxy = (boxes_xyxy / scale).clamp(min=0.0, max=1.0)
        boxes = box_convert(boxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh")
        return ObjectDetectionBatchOutput(logits=logits, boxes=boxes)

    def forward(self, images: Tensor) -> ObjectDetectionBatchOutput:
        """Run the model and return the raw graph outputs.

        The anchor decode is part of the exported graph. Top-k selection,
        thresholding, class-id remapping and rescaling to original image coordinates
        happen in :meth:`postprocess`.

        Args:
            images: Input tensor of shape (B, C, H, W).

        Returns:
            An :class:`ObjectDetectionBatchOutput` with raw logits and normalized
            ``cxcywh`` boxes relative to the model input.
        """
        feats = self.backbone(images)
        feats = self.neck(feats)
        cls_scores_list, bbox_preds_list = self.o2o_head(feats)
        return self.decode_o2o_outputs(
            cls_scores_list=cls_scores_list,
            bbox_preds_list=bbox_preds_list,
            input_size=(int(images.shape[-2]), int(images.shape[-1])),
        )

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: ObjectDetectionBatchOutput | Mapping[str, Tensor],
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        """Decode raw outputs into one prediction per image.

        Args:
            raw_outputs:
                Either an :class:`ObjectDetectionBatchOutput` as returned by
                :meth:`forward`, or a mapping with ``pred_logits`` and
                ``pred_boxes`` keys.
            metadata: Per-image metadata as returned by the preprocessor.
            threshold: Detections with a score <= threshold are discarded.

        Returns:
            A list with one :class:`ObjectDetectionPrediction` per input image, with
            boxes in original-image ``xyxy`` pixel coordinates.
        """
        if isinstance(raw_outputs, ObjectDetectionBatchOutput):
            raw = raw_outputs
        else:
            raw = ObjectDetectionBatchOutput(
                logits=raw_outputs["pred_logits"], boxes=raw_outputs["pred_boxes"]
            )
        return self.postprocessor.postprocess(
            raw=raw, metadata=metadata, threshold=threshold
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        threshold: float = 0.6,
    ) -> list[ObjectDetectionPrediction]:
        """Run inference on a batch of images and return per-image predictions.

        Args:
            images:
                Sequence of input images. Each can be a path, a PIL image, or a
                tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions
                with scores <= threshold are discarded.

        Returns:
            A list with one :class:`ObjectDetectionPrediction` per input image.
        """
        first_param = next(self.parameters())
        batch, metadata = self.preprocessor.preprocess(
            images, device=first_param.device, dtype=first_param.dtype
        )
        self._track_inference()
        if self.training or not self.is_deploy_mode:
            self.deploy()
        return self.postprocessor.postprocess(
            raw=self(batch), metadata=metadata, threshold=threshold
        )

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.6
    ) -> ObjectDetectionPrediction:
        """Run inference on a single image and return task-specific predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions with
                scores <= threshold are discarded.

        Returns:
            An :class:`ObjectDetectionPrediction` with ``labels`` of shape ``(N,)``,
            ``bboxes`` of shape ``(N, 4)`` in ``xyxy`` format, and ``scores`` of shape
            ``(N,)``.
        """
        return self.predict_batch([image], threshold=threshold)[0]

    @torch.no_grad()
    def predict_sahi(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.6,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
    ) -> ObjectDetectionPrediction:
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
            An :class:`ObjectDetectionPrediction` in original-image coordinates.
        """
        return self.predict_sahi_batch(
            [image],
            threshold=threshold,
            overlap=overlap,
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )[0]

    @torch.no_grad()
    def predict_sahi_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        threshold: float = 0.6,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
    ) -> list[ObjectDetectionPrediction]:
        """Run Slicing Aided Hyper Inference on a batch of images.

        All tiles of all images go through the model in a single forward pass. The
        postprocessor maps each image back to its own slice of the raw output.
        """
        sahi_config = ObjectDetectionSAHIConfig(
            overlap=overlap,
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )
        first_param = next(self.parameters())
        batch, metadata = self.preprocessor.preprocess(
            images,
            device=first_param.device,
            dtype=first_param.dtype,
            sahi_config=sahi_config,
        )
        self._track_inference()
        if self.training or not self.is_deploy_mode:
            self.deploy()
        return self.postprocessor.postprocess(
            raw=self(batch),
            metadata=metadata,
            threshold=threshold,
            sahi_config=sahi_config,
        )

    def verify_onnx_export_outputs(
        self,
        *,
        torch_outputs: BaseModelOutput,
        onnx_outputs: BaseModelOutput,
    ) -> None:
        if not isinstance(torch_outputs, ObjectDetectionBatchOutput) or not isinstance(
            onnx_outputs, ObjectDetectionBatchOutput
        ):
            raise TypeError(
                "PicoDet ONNX verification expects ObjectDetectionBatchOutput instances."
            )
        # The o2o peak filter keeps an anchor only if its score is exactly equal to
        # the max-pooled score of its neighborhood. Backend rounding can therefore
        # keep a slightly different set of anchors, and suppressed anchors carry a
        # large negative sentinel logit, so comparing raw logits is meaningless: a
        # single flipped anchor shifts the sum by the sentinel. Comparing sigmoid
        # scores bounds a differing anchor's contribution by its own score instead.
        # Summing over queries keeps the check independent of anchor ordering.
        for output_name in ("logits", "boxes"):
            output_model = getattr(torch_outputs, output_name)
            output_onnx = getattr(onnx_outputs, output_name)
            if output_onnx.is_floating_point():
                output_onnx = output_onnx.float()
            if output_name == "logits":
                output_model = output_model.sigmoid()
                output_onnx = output_onnx.sigmoid()
            output_model = output_model.sum(dim=1)
            output_onnx = output_onnx.sum(dim=1)

            def msg(s: str, output_name: str = output_name) -> str:
                return f'ONNX validation failed for output "{output_name}": {s}'

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
        shape_overrides: dict[str, tuple[int | None, ...]] | None = None,
    ) -> None:
        """Export the model to ONNX using its declared model I/O specification.

        The exported graph returns raw class logits of shape
        ``(batch_size, num_anchors, num_classes)`` and normalized ``cxcywh`` boxes of
        shape ``(batch_size, num_anchors, 4)``. The NMS-free o2o peak filter is part
        of the graph, but top-k selection, thresholding, class-id remapping, box
        rescaling and SAHI merging are intentionally kept outside it.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. Either "fp32" or "fp16".
            batch_size:
                Batch size for the ONNX input when ``dynamic_batch_size`` is False.
            dynamic_batch_size:
                If True, the ONNX graph will have a dynamic batch dimension for the
                input. If False, the batch dimension is fixed to `batch_size`.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to `torch.onnx.export`.
            num_channels:
                Number of input channels. If None, the value declared by the model
                input specification is used.
            shape_overrides:
                Reserved for compatibility with the shared ONNX export interface.
                Custom shape overrides are not supported for PicoDet.

        Raises:
            ValueError: If ``shape_overrides`` is not None.
        """
        if shape_overrides is not None:
            raise ValueError(
                "shape_overrides is not supported for PicoDet object detection."
            )

        super().export_onnx(
            out,
            precision=precision,
            batch_size=batch_size,
            dynamic_batch_size=dynamic_batch_size,
            opset_version=opset_version,
            simplify=simplify,
            verify=verify,
            format_args=format_args,
            shape_overrides=(
                {"images": (num_channels, None, None)}
                if num_channels is not None
                else None
            ),
        )

    @torch.no_grad()
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

        # The helper drives the ONNX export itself, so the requested precision has to
        # be forwarded explicitly. Without this an fp16 engine would be built from an
        # fp32 graph.
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
            fp32_attention_scores=False,
            strongly_typed=False,
            verbose=verbose,
        )
