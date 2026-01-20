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
from lightly_train._task_models.picodet_object_detection.pico_head import PicoHead
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

    def _forward_train(self, images: Tensor) -> dict[str, list[Tensor]]:
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
        return {"cls_scores": cls_scores, "bbox_preds": bbox_preds}

    def forward(
        self, images: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning final predictions for inference/ONNX.

        Args:
            images: Input tensor of shape (B, C, H, W).
            orig_target_size: Optional tensor of shape (B, 2) with (H, W) per image.

        Returns:
            Tuple of:
            - labels: Tensor of shape (B, N) with class indices.
            - boxes: Tensor of shape (B, N, 4) in xyxy format.
            - scores: Tensor of shape (B, N) with confidence scores.
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

        outputs = self._forward_train(images)
        result = self.postprocessor(
            cls_scores=[cs[:1] for cs in outputs["cls_scores"]],
            bbox_preds=[bp[:1] for bp in outputs["bbox_preds"]],
            original_size=(orig_h, orig_w),
            score_threshold=0.0,
        )

        max_detections = self.postprocessor.max_detections
        labels_out = torch.full(
            (1, max_detections),
            -1,
            device=images.device,
            dtype=torch.long,
        )
        boxes_out = torch.zeros(
            (1, max_detections, 4),
            device=images.device,
            dtype=result["bboxes"].dtype,
        )
        scores_out = torch.zeros(
            (1, max_detections),
            device=images.device,
            dtype=result["scores"].dtype,
        )

        # PicoDet postprocessing returns variable-length outputs, so we pad to
        # fixed shapes for ONNX; LTDETR already returns fixed-size tensors.
        labels = self.internal_class_to_class[result["labels"]]
        num_detections = labels.shape[0]
        labels_out[0, :num_detections] = labels
        boxes_out[0, :num_detections] = result["bboxes"]
        scores_out[0, :num_detections] = result["scores"]

        return labels_out, boxes_out, scores_out

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

        outputs = self._forward_train(x)
        results = self.postprocessor(
            cls_scores=outputs["cls_scores"],
            bbox_preds=outputs["bbox_preds"],
            original_size=(orig_h, orig_w),
            score_threshold=threshold,
        )
        labels = self.internal_class_to_class[results["labels"]]
        boxes = results["bboxes"]
        scores = results["scores"]
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
        The ONNX graph outputs labels, boxes, and scores in the resized input
        image space.

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
        output_names = ["labels", "boxes", "scores"]

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
