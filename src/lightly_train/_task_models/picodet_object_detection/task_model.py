#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Literal

import torch
from packaging import version
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train import _logging, _torch_testing
from lightly_train._commands import _warnings
from lightly_train._data import file_helpers
from lightly_train._export import tensorrt_helpers
from lightly_train._task_models.picodet_object_detection.csp_pan import CSPPAN
from lightly_train._task_models.picodet_object_detection.esnet import ESNet
from lightly_train._task_models.picodet_object_detection.pico_head import (
    PicoHead,
    distance2bbox,
)
from lightly_train._task_models.picodet_object_detection.postprocessor import (
    PicoDetPostProcessor,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

# Model configurations
_MODEL_CONFIGS = {
    "picodet/s-416": {
        "model_size": "s",
        "image_size": (416, 416),
        "stacked_convs": 2,
        "neck_out_channels": 96,
        "head_feat_channels": 96,
    },
    "picodet/l-416": {
        "model_size": "l",
        "image_size": (416, 416),
        "stacked_convs": 4,
        "neck_out_channels": 160,
        "head_feat_channels": 160,
    },
}


class PicoDetObjectDetection(TaskModel):
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
        image_normalize: dict[str, list[float]] | None = None,
        reg_max: int = 7,
        score_threshold: float = 0.025,
        iou_threshold: float = 0.6,
        max_detections: int = 100,
        load_weights: bool = True,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"load_weights"})

        self.model_name = model_name
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.classes = classes

        if classes is not None and len(classes) != num_classes:
            raise ValueError(
                "classes must have the same length as num_classes when provided."
            )

        internal_class_to_class = (
            list(range(num_classes)) if classes is None else list(classes.keys())
        )
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
        )

        config = _MODEL_CONFIGS.get(model_name)
        if config is None:
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                f"Available: {list(_MODEL_CONFIGS.keys())}"
            )

        model_size_raw = config["model_size"]
        stacked_convs_raw = config["stacked_convs"]
        neck_out_channels_raw = config["neck_out_channels"]
        head_feat_channels_raw = config["head_feat_channels"]
        if model_size_raw not in ("s", "m", "l"):
            raise ValueError(f"Invalid model_size: {model_size_raw}")
        if not isinstance(stacked_convs_raw, int):
            raise TypeError(f"stacked_convs must be int, got {type(stacked_convs_raw)}")
        if not isinstance(neck_out_channels_raw, int):
            raise TypeError(
                f"neck_out_channels must be int, got {type(neck_out_channels_raw)}"
            )
        if not isinstance(head_feat_channels_raw, int):
            raise TypeError(
                f"head_feat_channels must be int, got {type(head_feat_channels_raw)}"
            )
        model_size_typed: Literal["s", "m", "l"] = model_size_raw  # type: ignore[assignment]
        stacked_convs_typed: int = stacked_convs_raw
        neck_out_channels_typed: int = neck_out_channels_raw
        head_feat_channels_typed: int = head_feat_channels_raw

        self.backbone = ESNet(
            model_size=model_size_typed,
            out_indices=(2, 9, 12),  # C3, C4, C5
        )
        backbone_out_channels = self.backbone.out_channels

        self.neck = CSPPAN(
            in_channels=backbone_out_channels,
            out_channels=neck_out_channels_typed,
            kernel_size=5,
            num_features=4,  # P3, P4, P5, P6
            expansion=1.0,
            num_csp_blocks=1,
            use_depthwise=True,
        )

        self.head = PicoHead(
            in_channels=neck_out_channels_typed,
            num_classes=num_classes,
            feat_channels=head_feat_channels_typed,
            stacked_convs=stacked_convs_typed,
            kernel_size=5,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            share_cls_reg=True,
            use_depthwise=True,
        )
        self.o2o_head = PicoHead(
            in_channels=neck_out_channels_typed,
            num_classes=num_classes,
            feat_channels=head_feat_channels_typed,
            stacked_convs=stacked_convs_typed,
            kernel_size=5,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            share_cls_reg=True,
            use_depthwise=True,
        )

        self.postprocessor = PicoDetPostProcessor(
            num_classes=num_classes,
            reg_max=reg_max,
            strides=(8, 16, 32, 64),
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    @classmethod
    def list_model_names(cls) -> list[str]:
        """Return list of supported model names."""
        return list(_MODEL_CONFIGS.keys())

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        """Check if a model name is supported."""
        return model in _MODEL_CONFIGS

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

        if "internal_class_to_class" not in new_state_dict:
            new_state_dict["internal_class_to_class"] = (
                self.internal_class_to_class.detach().clone()
            )

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

    def _decode_o2o_predictions(
        self,
        *,
        cls_scores_list: list[Tensor],
        bbox_preds_list: list[Tensor],
        image_size: tuple[int, int],
        input_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        batch_size = cls_scores_list[0].shape[0]
        device = cls_scores_list[0].device
        decode_bbox_preds_pixel: list[Tensor] = []
        flatten_cls_preds: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(
            zip(cls_scores_list, bbox_preds_list)
        ):
            stride = self.o2o_head.strides[level_idx]
            _, _, h, w = cls_score.shape
            num_points = h * w

            y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
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
            decode_bbox_preds_pixel.append(decode_bbox_pred * stride)

            cls_pred_flat = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, self.num_classes
            )
            flatten_cls_preds.append(cls_pred_flat)

        boxes_xyxy = torch.cat(decode_bbox_preds_pixel, dim=1)
        cls_logits = torch.cat(flatten_cls_preds, dim=1)

        input_h, input_w = input_size
        orig_h, orig_w = image_size
        if (orig_h, orig_w) != (input_h, input_w):
            scale = boxes_xyxy.new_tensor(
                [orig_w / input_w, orig_h / input_h, orig_w / input_w, orig_h / input_h]
            )
            boxes_xyxy = boxes_xyxy * scale

        scale_limit = boxes_xyxy.new_tensor([orig_w, orig_h, orig_w, orig_h])
        boxes_xyxy = torch.min(boxes_xyxy, scale_limit).clamp(min=0)
        return boxes_xyxy, cls_logits

    def forward(
        self, images: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning o2o predictions for inference/ONNX.

        Args:
            images: Input tensor of shape (B, C, H, W).
            orig_target_size: Optional tensor of shape (B, 2) with (H, W) per image.

        Returns:
            Tuple of:
            - boxes_xyxy: Tensor of shape (B, N, 4) in xyxy pixel format.
            - obj_logits: Tensor of shape (B, N) with objectness logits.
            - cls_logits: Tensor of shape (B, N, C) with class logits.
        """
        if orig_target_size is None:
            orig_h, orig_w = images.shape[-2:]
        else:
            orig_target_size_ = orig_target_size.to(
                device=images.device, dtype=torch.int64
            )
            if orig_target_size_.ndim == 2:
                orig_target_size_ = orig_target_size_[0]
            orig_h, orig_w = int(orig_target_size_[0]), int(orig_target_size_[1])

        feats = self.backbone(images)
        feats = self.neck(feats)
        cls_scores_list, bbox_preds_list = self.o2o_head(feats)
        input_size = (int(images.shape[-2]), int(images.shape[-1]))
        boxes_xyxy, cls_logits = self._decode_o2o_predictions(
            cls_scores_list=cls_scores_list,
            bbox_preds_list=bbox_preds_list,
            image_size=(orig_h, orig_w),
            input_size=input_size,
        )
        obj_logits = cls_logits.max(dim=-1).values
        return boxes_xyxy, obj_logits, cls_logits

    @torch.no_grad()
    def predict(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.6,
    ) -> dict[str, Tensor]:
        """Run inference on a single image.

        Args:
            image: Input image as path, PIL image, or tensor (C, H, W).
            threshold: Score threshold for detections.

        Returns:
            Dictionary with:
            - labels: Tensor of shape (N,) with class indices.
            - bboxes: Tensor of shape (N, 4) with boxes in xyxy format.
            - scores: Tensor of shape (N,) with confidence scores.
        """
        self._track_inference()
        if self.training:
            self.eval()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)
        if self.image_normalize is not None:
            x = transforms_functional.normalize(
                x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
            )
        x = transforms_functional.resize(x, list(self.image_size))
        x = x.unsqueeze(0)

        feats = self.backbone(x)
        feats = self.neck(feats)
        cls_scores_list, bbox_preds_list = self.o2o_head(feats)
        boxes_xyxy, cls_logits = self._decode_o2o_predictions(
            cls_scores_list=cls_scores_list,
            bbox_preds_list=bbox_preds_list,
            image_size=(orig_h, orig_w),
            input_size=tuple(self.image_size),
        )
        boxes = boxes_xyxy[0]
        internal_labels = cls_logits[0].argmax(dim=-1)
        cls_for_label = cls_logits[0].gather(1, internal_labels.unsqueeze(1)).squeeze(1)
        scores = torch.sigmoid(cls_for_label)
        labels = self.internal_class_to_class[internal_labels]
        if threshold > 0:
            keep = scores >= threshold
            labels = labels[keep]
            boxes = boxes[keep]
            scores = scores[keep]
        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    @torch.no_grad()
    def export_onnx(
        self,
        out: PathLike,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
        num_channels: int | None = None,
    ) -> None:
        """Exports the model to ONNX for inference.

        The export uses a dummy input of shape (1, C, H, W) where C is inferred
        from the first model parameter and (H, W) come from `self.image_size`.
        The ONNX graph outputs decoded boxes plus raw objectness and class logits:
        boxes_xyxy, obj_logit, cls_logits.

        Optionally simplifies the exported model in-place using onnxslim and
        verifies numerical closeness against a float32 CPU reference via
        ONNX Runtime.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. Either "auto", "fp32", or "fp16". "auto"
                uses the model's current precision.
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
                Number of input channels. If None, will be inferred.
        """
        _warnings.filter_export_warnings()
        _logging.set_up_console_logging()

        self.eval()
        self.postprocessor.deploy()

        first_parameter = next(self.parameters())
        model_device = first_parameter.device
        dtype = first_parameter.dtype

        if precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision != "auto":
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'auto', 'fp32', 'fp16'."
            )

        self.to(dtype)
        model_device = next(self.parameters()).device

        if num_channels is None:
            if self.image_normalize is not None:
                num_channels = len(self.image_normalize["mean"])
                logger.info(
                    f"Inferred num_channels={num_channels} from image_normalize."
                )
            else:
                for module in self.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        num_channels = module.in_channels
                        logger.info(
                            f"Inferred num_channels={num_channels} from first Conv. layer."
                        )
                        break
                if num_channels is None:
                    logger.error(
                        "Could not infer num_channels. Please provide it explicitly."
                    )
                    raise ValueError(
                        "num_channels must be provided for ONNX export if it cannot be inferred."
                    )

        dummy_input = torch.randn(
            1,
            num_channels,
            self.image_size[0],
            self.image_size[1],
            requires_grad=False,
            device=model_device,
            dtype=dtype,
        )

        input_names = ["images"]
        output_names = ["boxes_xyxy", "obj_logit", "cls_logits"]

        # Older torch.onnx.export versions don't accept the "dynamo" kwarg.
        export_kwargs: dict[str, Any] = {
            "input_names": input_names,
            "output_names": output_names,
            "opset_version": opset_version,
            "dynamic_axes": {"images": {0: "N"}},
            **(format_args or {}),
        }
        torch_version = version.parse(torch.__version__.split("+", 1)[0])
        if torch_version >= version.parse("2.2.0"):
            export_kwargs["dynamo"] = False

        torch.onnx.export(
            self,
            (dummy_input,),
            str(out),
            **export_kwargs,
        )

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                str(out),
                output_model=out,
                skip_optimizations=["constant_folding"],
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(out, full_check=True)

            reference_model = deepcopy(self).cpu().to(torch.float32).eval()
            reference_outputs = reference_model(
                dummy_input.cpu().to(torch.float32),
            )

            session = ort.InferenceSession(out)
            input_feed = {
                "images": dummy_input.cpu().numpy(),
            }
            outputs_onnx = session.run(output_names=None, input_feed=input_feed)
            outputs_onnx = tuple(torch.from_numpy(y) for y in outputs_onnx)

            if len(outputs_onnx) != len(reference_outputs):
                raise AssertionError(
                    f"Number of onnx outputs should be {len(reference_outputs)} but is {len(outputs_onnx)}"
                )
            for output_onnx, output_model, output_name in zip(
                outputs_onnx, reference_outputs, output_names
            ):

                def msg(s: str) -> str:
                    return f'ONNX validation failed for output "{output_name}": {s}'

                if output_model.is_floating_point:
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
                else:
                    _torch_testing.assert_most_equal(
                        output_onnx,
                        output_model,
                        msg=msg,
                    )

        logger.info(f"Successfully exported ONNX model to '{out}'")

    @torch.no_grad()
    def export_tensorrt(
        self,
        out: PathLike,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
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
                "auto", "fp32", or "fp16". "auto" uses the model's current precision.
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

        tensorrt_helpers.export_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=model_dtype,
            onnx_args=onnx_args,
            max_batchsize=max_batchsize,
            opt_batchsize=opt_batchsize,
            min_batchsize=min_batchsize,
            verbose=verbose,
        )
