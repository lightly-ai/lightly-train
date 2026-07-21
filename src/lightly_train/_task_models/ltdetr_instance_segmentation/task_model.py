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
from copy import deepcopy
from typing import Any, Literal

import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.ops import masks_to_boxes
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self, override

from lightly_train import _logging, _torch_testing
from lightly_train._commands import _warnings
from lightly_train._data import file_helpers
from lightly_train._export import tensorrt_helpers
from lightly_train._export.onnx_helpers import (
    fix_topological_order,
    remove_redundant_casts,
)
from lightly_train._models import package_helpers
from lightly_train._models.ecvit.ecvit import ECViTModelWrapper
from lightly_train._models.ecvit.ecvit_package import EDGE_CRAFTER_PACKAGE
from lightly_train._task_models.instance_segmentation_components.edgecrafter_decoder import (
    ECSegTransformer,
)
from lightly_train._task_models.instance_segmentation_components.edgecrafter_postprocessor import (
    ECSegPostProcessor,
)
from lightly_train._task_models.ltdetr_instance_segmentation.config import (
    LTDETR_SEG_MODEL_REGISTRY,
    SegmentorConfig,
)
from lightly_train._task_models.ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class LTDETRInstanceSegmentation(TaskModel):
    model_suffix = "ltdetr-seg"

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
        load_weights: bool = True,
    ) -> None:
        """Create an LTDETR instance segmentation task model.

        Args:
            model_name:
                The model name. For example ``"ltdetrv2-s"`` or
                ``"edgecrafter/ecvitt-ltdetr"``.
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
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        # Store init_args for checkpointing.
        super().__init__(init_args=locals(), ignore_args={"load_weights"})

        config: SegmentorConfig = LTDETR_SEG_MODEL_REGISTRY.get(alias=model_name)()

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
        # This list maps the internal class id to the class id in `classes`.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )
        self.included_classes: dict[int, str] = {
            internal_class_id: class_name
            for internal_class_id, class_name in enumerate(self.classes.values())
        }

        self.image_normalize = image_normalize
        self._expected_input_channels = 3

        # Build backbone model args: start from config defaults, then apply overrides.
        if backbone_args is not None:
            raise ValueError(
                "backbone_args are not supported for ECViT instance segmentation."
            )
        if backbone_weights is not None:
            raise ValueError(
                "backbone_weights are not supported for ECViT instance segmentation."
            )

        backbone = EDGE_CRAFTER_PACKAGE.get_model(
            model_name=short_backbone,
            model_args=None,
            load_weights=load_weights,
        )
        assert isinstance(backbone, ECViTModelWrapper)
        self.backbone: ECViTBackboneWrapper = ECViTBackboneWrapper(
            model_wrapper=backbone
        )

        self.encoder: HybridEncoder = HybridEncoder(
            **config.hybrid_encoder.model_dump()
        )

        transformer_cfg = config.transformer.model_dump()
        transformer_cfg["num_classes"] = len(self.classes)
        self.decoder: ECSegTransformer = ECSegTransformer(  # type: ignore[no-untyped-call]
            **transformer_cfg,
            eval_spatial_size=self.image_size,
        )

        postprocessor_cfg = config.ecseg_postprocessor.model_dump()
        postprocessor_cfg["num_classes"] = len(self.classes)
        self.postprocessor: ECSegPostProcessor = ECSegPostProcessor(**postprocessor_cfg)

        if self.backbone_freeze:
            self.freeze_backbone()

    @override
    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        return model in LTDETR_SEG_MODEL_REGISTRY.list_aliases()

    @classmethod
    def list_model_names(cls) -> list[str]:
        return LTDETR_SEG_MODEL_REGISTRY.list_model_names()

    @classmethod
    def parse_model_name(cls, model_name: str) -> dict[str, str]:
        """Resolve a registered model alias into its package/backbone parts."""
        try:
            config = LTDETR_SEG_MODEL_REGISTRY.get(alias=model_name)()
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
        return ["labels", "boxes", "masks", "scores"]

    def forward_backend(self, x: Tensor) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)

        # Don't pass ``spatial_feat`` so the decoder uses its projected feature
        # ``proj_feats[0]``. For ViT/ECViT presets the decoder's ``input_proj`` is
        # ``Identity`` (encoder and decoder share ``hidden_dim``), so this matches
        # EdgeCrafter's ``spatial_feat = x[0]``. For ConvNeXt presets the encoder
        # emits more channels than the decoder ``hidden_dim``; using the projected
        # feature gives the mask head the channel count it expects.
        return self.decoder(feats=x)

    def forward(
        self, x: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if orig_target_size is None:
            h, w = x.shape[-2:]
            orig_target_size_ = torch.tensor([[w, h]]).to(x.device)
        else:
            orig_target_size_ = orig_target_size[:, [1, 0]].to(
                device=x.device,
                dtype=torch.int64,
            )

        x = self.forward_backend(x)

        result: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor, Tensor] = (
            self.postprocessor(x, orig_target_size_)
        )
        # Postprocessor must be in deploy mode at this point. It returns only tuples
        # during deploy mode.
        assert isinstance(result, tuple) and len(result) == 4
        labels, boxes, scores, masks = result
        labels = self.internal_class_to_class[labels]
        return (labels, boxes, masks, scores)

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        # See ``forward_backend``: omit ``spatial_feat`` so the decoder uses its
        # projected feature ``proj_feats[0]`` (Identity for ViT/ECViT presets).
        x = self.decoder(feats=x, targets=targets)
        return x

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: Any | dict[str, Tensor],
        metadata: Sequence[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Tensor]]:
        if not isinstance(raw_outputs, dict):
            raise ValueError(
                f"Expected raw_outputs to be a dict, got {type(raw_outputs).__name__}."
            )

        device = next(self.parameters()).device
        orig_target_size = torch.tensor(
            [[m["orig_w"], m["orig_h"]] for m in metadata],
            dtype=torch.int64,
            device=device,
        )
        postprocessor_out: tuple[Tensor, Tensor, Tensor, Tensor] = self.postprocessor(
            raw_outputs, orig_target_size
        )
        out: list[dict[str, Tensor]] = []
        labels_batch, boxes_batch, scores_batch, masks_batch = postprocessor_out

        labels_batch = self.internal_class_to_class[labels_batch]
        for i, meta in enumerate(metadata):
            keep = scores_batch[i] > threshold
            # The deploy postprocessor returns raw mask logits at the mask-head
            # resolution (image_size // downsample_ratio). Interpolate them back
            # to the original image size and binarize, matching the
            # postprocessor's non-deploy branch and the instance-segmentation
            # `predict` contract (boolean masks of shape (N, orig_h, orig_w)).
            masks = masks_batch[i][keep]
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(int(meta["orig_h"]), int(meta["orig_w"])),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            out.append(
                {
                    "labels": labels_batch[i][keep],
                    "bboxes": boxes_batch[i][keep],
                    "masks": masks > 0.0,
                    "scores": scores_batch[i][keep],
                }
            )
        return out

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def deploy(self) -> Self:
        self.eval()
        self.postprocessor.deploy()  # type: ignore[no-untyped-call]
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        return self

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
        """Exports the model to ONNX for inference.

        The export uses a dummy input of shape ``(batch_size, C, H, W)`` where
        ``C`` is inferred and ``(H, W)`` come from ``self.image_size``. It also
        exports ``orig_target_size`` as a second input of shape ``(batch_size, 2)``
        in ``(H, W)`` format so boxes are scaled to caller-provided image sizes.

        Optionally simplifies the exported model in-place using onnxslim and
        verifies numerical closeness against a float32 CPU reference via
        ONNX Runtime.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. Either "fp32", or "fp16".
            batch_size:
                Batch size for the ONNX input.
            dynamic_batch_size:
                If True, the ONNX graph will have a dynamic batch dimension for the
                inputs. If False, the batch dimension is fixed to ``batch_size``.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to ``torch.onnx.export``.
            num_channels:
                Number of input channels. If None, will be inferred.
        """
        _warnings.filter_export_warnings()
        _logging.set_up_console_logging()

        self.eval()

        if precision not in ("fp32", "fp16"):
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'fp32', 'fp16'."
            )

        # Always trace in fp32 to avoid dtype mismatches in the decoder's
        # autocast(enabled=False) blocks. fp16 conversion is applied
        # post-export via onnxruntime.transformers.
        self.to(torch.float32)
        self.deploy()
        model_device = next(self.parameters()).device

        # Infer num_channels if not provided. The model always consumes
        # ``self._expected_input_channels`` channels (grayscale inputs are
        # expanded to this count before batching), so use it rather than the
        # normalization stats, whose length may not match the backbone's input.
        if num_channels is None:
            num_channels = self._expected_input_channels
            logger.info(
                f"Inferred num_channels={num_channels} from the model's expected "
                "input channels."
            )

        if dynamic_batch_size:
            batch_size = 2
            dynamic_axes = {
                "images": {0: "N"},
                "orig_target_size": {0: "N"},
            }
        else:
            dynamic_axes = None

        dummy_input = torch.randn(
            batch_size,
            num_channels,
            self.image_size[0],
            self.image_size[1],
            requires_grad=False,
            device=model_device,
            dtype=torch.float32,
        )
        dummy_orig_target_size = torch.tensor(
            [self.image_size],
            device=model_device,
            dtype=torch.int64,
        ).repeat(batch_size, 1)

        input_names = ["images", "orig_target_size"]
        output_names = self.get_export_output_names()

        torch.onnx.export(
            self,
            (dummy_input, dummy_orig_target_size),
            str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes=dynamic_axes,
            **(format_args or {}),
        )

        if precision == "fp16":
            # convert_float_to_float16 creates nodes with duplicate names. In order to avoid downstream issues
            # we require simplify to be True, as this correctly renames nodes.
            if not simplify:
                raise ValueError("fp16 precision requires simplify=True.")

            import onnx
            from onnxruntime.transformers import float16 as ort_float16

            model_onnx = onnx.load(str(out))
            # If the input to Softmax are too large the output of Softmax will be NaN values. Therefore we run
            #  the Softmax computation in fp32. The nodes before Softmax are always MatMul.
            # TODO (simon, 05/26) Ideally we would only block operators were a Matmul directly feeds into a Softmax.
            op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + [
                "Softmax",
                "MatMul",
            ]
            model_fp16 = ort_float16.convert_float_to_float16(
                model_onnx, op_block_list=op_block_list
            )
            # Using the op blocklist on a graph that looks like Softmax -> MatMul creates a graph that looks like
            #  Cast32 -> MatMul -> Cast16 -> Cast32 -> Softmax -> Cast16. Therefore, we need to remove the middle
            #  Cast16 -> Cast32.
            remove_redundant_casts(model_fp16)
            fix_topological_order(model_fp16)
            onnx.save(model_fp16, str(out))

        if simplify:
            import onnxslim  # type: ignore[import-not-found,import-untyped]

            onnxslim.slim(
                str(out),
                output_model=out,
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(out, full_check=True)

            providers = ort.get_available_providers()
            if precision == "fp16" and "CUDAExecutionProvider" not in providers:
                logger.warning(
                    "Skipping ONNX runtime verification for fp16 model because "
                    "CUDAExecutionProvider is not available in onnxruntime. "
                    "Install onnxruntime-gpu to enable full verification."
                )
            else:
                # Always run the reference input in float32 and on cpu for consistency.
                reference_model = deepcopy(self).cpu().to(torch.float32).eval()
                reference_model.deploy()
                reference_outputs: tuple[Tensor, ...] = reference_model(
                    dummy_input.cpu().to(torch.float32),
                    dummy_orig_target_size.cpu(),
                )

                # Get outputs from the ONNX model. Load from bytes to avoid
                # ORT errors about missing external data when weights are inline.
                with open(out, "rb") as f:
                    session = ort.InferenceSession(f.read())
                onnx_input = dummy_input.cpu()
                if precision == "fp16":
                    onnx_input = onnx_input.half()
                input_feed = {
                    "images": onnx_input.numpy(),
                    "orig_target_size": dummy_orig_target_size.cpu().numpy(),
                }
                outputs_onnx = session.run(output_names=None, input_feed=input_feed)
                outputs_onnx = tuple(torch.from_numpy(y) for y in outputs_onnx)

                # Verify that the outputs from both models are close.
                if len(outputs_onnx) != len(reference_outputs):
                    raise AssertionError(
                        f"Number of onnx outputs should be {len(reference_outputs)} but is {len(outputs_onnx)}"
                    )
                for output_onnx, output_model, output_name in zip(
                    outputs_onnx, reference_outputs, output_names
                ):

                    def msg(s: str) -> str:
                        return f'ONNX validation failed for output "{output_name}": {s}'

                    # Due to the presence of top-k operations in the model, the
                    # outputs may be in a different order but still valid. To
                    # compare in an order-invariant way we reduce along the query
                    # dimension before comparing.
                    # TODO(yutong, 07/2026): Reducing each output independently
                    # only checks per-field marginals, so it cannot detect an
                    # export that swaps labels or masks between boxes (the sums
                    # and label multiset stay identical). Match detections per
                    # image by box location and compare the full
                    # (label, box, mask, score) tuples together instead.
                    if output_model.is_floating_point():
                        # Float outputs (boxes, masks, scores): sum over the query
                        # dimension. Convert the ONNX output to fp32 first to avoid
                        # overflow when summing in fp16.
                        output_model = output_model.sum(dim=1)
                        output_onnx = output_onnx.float().sum(dim=1)
                        # Absolute and relative tolerances are a bit arbitrary and taken from here:
                        # https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_core.py#L1611-L1618
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
                        # Integer outputs (labels): compare as order-invariant
                        # multisets by sorting along the query dimension. Summing
                        # would let an incorrect set such as [0, 2] match [1, 1].
                        # Require an exact match (min_fraction=1.0): once sorted
                        # there is no ordering slack to allow, and the default
                        # 0.99 would let a few wrong class labels pass silently.
                        output_model = output_model.sort(dim=1).values
                        output_onnx = output_onnx.sort(dim=1).values
                        _torch_testing.assert_most_equal(
                            output_onnx,
                            output_model,
                            min_fraction=1.0,
                            msg=msg,
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

    def preprocess_image(
        self, image: PathLike | PILImage | Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype

        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        # Expand grayscale to the expected channel count so images can be stacked.
        # TODO(Nauryzbay, 05/26): Revisit grayscale handling — the implicit
        # 1-channel expansion is a convenience inherited from RGB-only models.
        expected_c = self._expected_input_channels
        if x.shape[-3] == 1 and expected_c > 1:
            x = x.expand(expected_c, -1, -1)
        elif x.shape[-3] != expected_c:
            raise ValueError(
                f"Image has {x.shape[-3]} channels but model expects {expected_c}."
            )

        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        x = transforms_functional.resize(x, self.image_size)
        return x, {"orig_h": image_h, "orig_w": image_w}

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        if self.image_normalize is not None:
            batch = transforms_functional.normalize(
                batch,
                mean=list(self.image_normalize["mean"]),
                std=list(self.image_normalize["std"]),
            )
        return batch

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
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()
        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for image in images:
            x, meta = self.preprocess_image(image)
            tensors.append(x)
            metadata.append(meta)
        batch = torch.stack(tensors, dim=0)
        batch = self.preprocess_batch(batch)
        raw = self.forward_backend(batch)
        return self.postprocess(raw, metadata, threshold=threshold)

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
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()
        x, metadata = self.preprocess_image(image)
        batch = self.preprocess_batch(x.unsqueeze(0))
        raw = self.forward_backend(batch)
        return self.postprocess(raw, [metadata], threshold=threshold)[0]

    @torch.no_grad()
    def predict_sahi(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.8,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.5,
        global_local_iou_threshold: float = 0.5,
        batch_size: int | None = None,
    ) -> dict[str, Tensor]:
        """Run Slicing Aided Hyper Inference (SAHI) for instance segmentation.

        The image is converted to a tensor, then:

        - A resized full-image "global" prediction is computed.
        - The image is tiled into overlapping crops of size ``self.image_size``.
        - Each tile is run through the model (in batches of ``batch_size``) and its
          masks are stitched back into the original image coordinates.
        - Global and tile predictions are merged using mask NMS and a global/local
          consistency heuristic. NMS is only applied on tile predictions; the
          heuristic discards tile predictions that heavily overlap global
          predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape
                (C, H, W).
            threshold:
                Score threshold for filtering low-confidence predictions.
            overlap:
                Fractional overlap between tiles in [0, 1). 0.0 means no overlap.
            nms_iou_threshold:
                Mask IoU threshold used for non-maximum suppression when merging
                predictions from tiles. A lower value yields fewer predictions.
            global_local_iou_threshold:
                Mask IoU above which a tile prediction is discarded because it
                matches a global prediction of the same class. A lower value yields
                fewer predictions.
            batch_size:
                Number of tiles to run through the model at once. If None, all tiles
                are processed in a single batch. Must be a positive integer.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with boxes in (x_min, y_min,
                    x_max, y_max) at the original image resolution. Boxes are the
                    tight bounding boxes of the merged masks.
                - "masks": Tensor of shape (N, H, W) with binary masks at the
                    original image resolution.
                - "scores": Tensor of shape (N,) with confidence scores.
        """
        self._track_inference()
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()

        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype

        x = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = x.shape[-2:]
        tile_h, tile_w = self.image_size

        # Scale to float once up front. Normalization happens per batch in
        # ``preprocess_batch`` (unlike the EoMT models, whose ``preprocess_batch``
        # only pads), so we must not normalize here to avoid double-normalizing.
        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)

        # Global full-image prediction: plain resize to the model input size, then
        # the regular predict pipeline. ``postprocess`` interpolates masks back to
        # the original resolution and binarizes them.
        x_global = transforms_functional.resize(x, self.image_size)
        global_batch = self.preprocess_batch(x_global.unsqueeze(0))
        global_raw = self.forward_backend(global_batch)
        pred_global = self.postprocess(
            global_raw,
            [{"orig_h": orig_h, "orig_w": orig_w}],
            threshold=threshold,
        )[0]

        tiles, coordinates = tiling_utils.tile_image(
            image=x,
            overlap=overlap,
            tile_size=self.image_size,
            padding_mode="pad",
        )

        if batch_size is None:
            batch_size = len(tiles)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        all_labels: list[Tensor] = []
        all_masks: list[Tensor] = []
        all_scores: list[Tensor] = []

        for start in range(0, len(tiles), batch_size):
            end = min(start + batch_size, len(tiles))
            tile_batch = self.preprocess_batch(tiles[start:end])
            coordinate_batch = coordinates[start:end]

            raw = self.forward_backend(tile_batch)
            metadata: list[dict[str, Any]] = []
            valid_sizes: list[tuple[int, int]] = []
            for coordinate in coordinate_batch:
                x_start = int(coordinate[0].item())
                y_start = int(coordinate[1].item())
                valid_h = min(tile_h, orig_h - y_start)
                valid_w = min(tile_w, orig_w - x_start)
                valid_sizes.append((valid_h, valid_w))
                metadata.append({"orig_h": tile_h, "orig_w": tile_w})

            predictions = self.postprocess(raw, metadata, threshold=threshold)
            for prediction, coordinate, valid_size in zip(
                predictions, coordinate_batch, valid_sizes
            ):
                labels = prediction["labels"]
                masks = prediction["masks"]
                scores = prediction["scores"]
                if labels.numel() == 0:
                    continue

                x_start = int(coordinate[0].item())
                y_start = int(coordinate[1].item())
                valid_h, valid_w = valid_size
                masks = masks[:, :valid_h, :valid_w]
                full_masks = masks.new_zeros(
                    (masks.shape[0], orig_h, orig_w), dtype=torch.bool
                )
                full_masks[
                    :, y_start : y_start + valid_h, x_start : x_start + valid_w
                ] = masks

                all_labels.append(labels)
                all_masks.append(full_masks)
                all_scores.append(scores)

        if len(all_labels) > 0:
            labels_tiles = torch.cat(all_labels, dim=0)
            masks_tiles = torch.cat(all_masks, dim=0)
            scores_tiles = torch.cat(all_scores, dim=0)
        else:
            labels_tiles = torch.empty(0, dtype=torch.long, device=device)
            masks_tiles = torch.empty(
                (0, orig_h, orig_w), dtype=torch.bool, device=device
            )
            scores_tiles = torch.empty(0, dtype=dtype, device=device)

        labels, masks, scores = tiling_utils.combine_instance_segmentation_tiles(
            pred_global={
                "labels": pred_global["labels"],
                "masks": pred_global["masks"],
                "scores": pred_global["scores"],
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )

        # Drop instances whose binarized mask has no positive pixels. Such masks
        # can survive the score threshold (the filter is on score, not mask
        # occupancy) and would make ``masks_to_boxes`` raise on the empty
        # reduction.
        if masks.numel() > 0:
            non_empty = masks.flatten(start_dim=1).any(dim=1)
            labels = labels[non_empty]
            masks = masks[non_empty]
            scores = scores[non_empty]

        # ``combine_instance_segmentation_tiles`` returns only labels/masks/scores.
        # Derive tight bboxes from the merged masks so the output matches the
        # ``predict`` contract and stays consistent with the post-NMS masks.
        bboxes = (
            masks_to_boxes(masks)
            if masks.numel() > 0
            else masks.new_zeros((0, 4), dtype=torch.float32)
        )

        return {
            "labels": labels,
            "bboxes": bboxes,
            "masks": masks,
            "scores": scores,
        }
