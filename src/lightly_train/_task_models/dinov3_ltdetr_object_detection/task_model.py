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
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train import _logging, _torch_testing
from lightly_train._commands import _warnings
from lightly_train._data import file_helpers
from lightly_train._export import tensorrt_helpers
from lightly_train._export.onnx_helpers import (
    fix_topological_order,
    remove_redundant_casts,
)
from lightly_train._models import package_helpers
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.ecvit.ecvit import ECViTModelWrapper
from lightly_train._models.ecvit.ecvit_package import EDGE_CRAFTER_PACKAGE
from lightly_train._task_models.dinov3_ltdetr.task_model import _DINOv3LTDETRBase
from lightly_train._task_models.dinov3_ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_convnext_wrapper import (
    DINOv3ConvNextWrapper,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_vit_wrapper import (
    DINOv3STAs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetr_postprocessor import (
    RTDETRPostProcessor,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class DINOv3LTDETRObjectDetection(_DINOv3LTDETRBase):
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
        """Create a DINOv3 LTDETR task model.

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
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        # Bypass _DINOv3LTDETRBase.__init__ (old config system) and call
        # TaskModel.__init__ directly to store init_args for checkpointing.
        super(_DINOv3LTDETRBase, self).__init__(
            init_args=locals(), ignore_args={"load_weights"}
        )

        config = LTDETR_MODEL_REGISTRY.get(alias=model_name)()

        package_name, short_backbone = package_helpers.parse_model_name(
            config.backbone_name
        )
        self.model_name = f"{config.backbone_name}-ltdetr"
        self.image_size = image_size
        self.classes = classes
        self.backbone_freeze = backbone_freeze

        if backbone_freeze:
            config.backbone_wrapper.finetune = False

        # Use the config's baked-in patch_size unless the caller overrides it.
        patch_size = patch_size or config.backbone_args.get("patch_size")
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

        # Resolve the backbone's expected input channel count.
        # backbone_args["in_chans"] overrides image_normalize, which overrides 3.
        if backbone_args is not None and "in_chans" in backbone_args:
            self._expected_input_channels: int = backbone_args["in_chans"]
        elif self.image_normalize is not None:
            self._expected_input_channels = len(self.image_normalize["mean"])
        else:
            self._expected_input_channels = 3

        # Build backbone model args: start from config defaults, then apply overrides.
        backbone_model_args: dict[str, Any] = dict(config.backbone_args)
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)
        if backbone_weights is not None:
            backbone_model_args["weights"] = str(backbone_weights)

        get_model_kwargs = {}
        if self.image_normalize is not None:
            get_model_kwargs["num_input_channels"] = len(self.image_normalize["mean"])

        package = package_helpers.get_package(package_name)

        # ECViT lives in its own package and rejects model_args entirely.
        if package_name == EDGE_CRAFTER_PACKAGE.name:
            backbone: ConvNeXt | DinoVisionTransformer | ECViTModelWrapper = (
                package.get_model(
                    model_name=short_backbone,
                    model_args=None,
                    load_weights=load_weights,
                )
            )
        else:
            backbone = package.get_model(
                model_name=short_backbone,
                model_args=backbone_model_args,
                load_weights=load_weights,
                **get_model_kwargs,
            )
        assert isinstance(
            backbone, (ConvNeXt, DinoVisionTransformer, ECViTModelWrapper)
        )

        self.backbone: DINOv3STAs | DINOv3ConvNextWrapper | ECViTBackboneWrapper

        if isinstance(backbone, ECViTModelWrapper):
            self.backbone = ECViTBackboneWrapper(model_wrapper=backbone)
        elif isinstance(backbone, DinoVisionTransformer):
            # TODO(Guarin, 02/26): Improve how mask tokens are handled for fine-tuning.
            backbone.mask_token.requires_grad = False  # type: ignore
            vit_model_wrapper = DINOv3ViTModelWrapper(backbone)
            self.backbone = DINOv3STAs(
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

        transformer_cfg = config.transformer.model_dump()
        transformer_cfg["num_classes"] = len(self.classes)
        if config.transformer.decoder_name == "rtdetrv2":
            self.decoder: RTDETRTransformerv2 | DFINETransformer = RTDETRTransformerv2(
                **transformer_cfg, eval_spatial_size=self.image_size
            )
        else:
            self.decoder = DFINETransformer(
                **transformer_cfg, eval_spatial_size=self.image_size
            )

        postprocessor_cfg = config.rtdetr_postprocessor.model_dump()
        postprocessor_cfg["num_classes"] = len(self.classes)
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_cfg
        )

        if self.backbone_freeze:
            self.freeze_backbone()

    def get_export_output_names(self) -> list[str]:
        return ["labels", "boxes", "scores"]

    def forward_backend(self, x: Tensor) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)

    def forward(
        self, x: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Function used for ONNX export
        if orig_target_size is None:
            h, w = x.shape[-2:]
            orig_target_size_ = torch.tensor([[w, h]]).to(x.device)
        else:
            # Flip from (H, W) to (W, H).
            orig_target_size = orig_target_size[:, [1, 0]]

            # Move to device.
            orig_target_size_ = orig_target_size.to(device=x.device, dtype=torch.int64)

        # Forward the image through the model.
        x = self.forward_backend(x)

        result: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor] = (
            self.postprocessor(x, orig_target_size_)
        )
        # Postprocessor must be in deploy mode at this point. It returns only tuples
        # during deploy mode.
        assert isinstance(result, tuple)
        labels, boxes, scores = result
        labels = self.internal_class_to_class[labels]
        return (labels, boxes, scores)

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
        if not isinstance(raw_outputs, dict):
            raise ValueError(
                f"Expected raw_outputs to be a dict, got {type(raw_outputs).__name__}."
            )
        device = next(self.parameters()).device
        # Postprocessor expects (W, H) per image.
        orig_target_size = torch.tensor(
            [[m["orig_w"], m["orig_h"]] for m in metadata],
            dtype=torch.int64,
            device=device,
        )
        postprocessor_out: tuple[Tensor, Tensor, Tensor] = self.postprocessor(
            raw_outputs, orig_target_size
        )
        out: list[dict[str, Tensor]] = []
        labels_batch, boxes_batch, scores_batch = postprocessor_out

        labels_batch = self.internal_class_to_class[labels_batch]
        for i in range(len(metadata)):
            keep = scores_batch[i] > threshold
            out.append(
                {
                    "labels": labels_batch[i][keep],
                    "bboxes": boxes_batch[i][keep],
                    "scores": scores_batch[i][keep],
                }
            )
        return out

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

        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)

        # Tile the image.
        tiles, tiles_coordinates = tiling_utils.tile_image(x, overlap, self.image_size)

        # Prepare the full image tile
        h, w = x.shape[-2:]
        x = transforms_functional.resize(x, self.image_size)
        x = x.unsqueeze(0)
        tiles = torch.cat([x, tiles], dim=0)

        # Normalize the tiles and the image together.
        tiles = transforms_functional.to_dtype(tiles, dtype=torch.float32, scale=True)

        # Normalize the tiles.
        if self.image_normalize is not None:
            tiles = transforms_functional.normalize(
                tiles,
                mean=self.image_normalize["mean"],
                std=self.image_normalize["std"],
            )

        # Prepare the image/tiles sizes.
        orig_target_sizes = torch.tensor([self.image_size], device=device).repeat(
            len(tiles), 1
        )
        orig_target_sizes[0, 0] = h
        orig_target_sizes[0, 1] = w

        # Feed the tiles in parallel to the model.
        labels, boxes, scores = self(tiles, orig_target_size=orig_target_sizes)

        # Add coordinates of the tiles to the boxes.
        tiles_coordinates = (
            tiles_coordinates.repeat(1, 2).unsqueeze(1).expand(-1, boxes.shape[1], -1)
        )
        boxes[1:] += tiles_coordinates

        # Reorganize the predictions.
        boxes_global = boxes[0].view(-1, 4)
        boxes_tiles = boxes[1:].view(-1, 4)
        labels_global = labels[0].flatten()
        labels_tiles = labels[1:].flatten()
        scores_global = scores[0].flatten()
        scores_tiles = scores[1:].flatten()

        # Discard low-confidence predictions.
        keep_global = scores_global > threshold
        keep_tiles = scores_tiles > threshold

        # Combine global and tiles predictions.
        labels, boxes, scores = tiling_utils.combine_object_detection_tiles(
            pred_global={
                "labels": labels_global[keep_global],
                "bboxes": boxes_global[keep_global],
                "scores": scores_global[keep_global],
            },
            pred_tiles={
                "labels": labels_tiles[keep_tiles],
                "bboxes": boxes_tiles[keep_tiles],
                "scores": scores_tiles[keep_tiles],
            },
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )

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

        The export uses a dummy input of shape (batch_size, C, H, W) where C is
        inferred from the first model parameter and (H, W) come from
        `self.image_size`. If `dynamic_batch_size` is True, the ONNX graph will
        have a dynamic batch dimension for the input. The graph output names are provided by the concrete task model.

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
                Number of input channels. If None, will be inferred.

        Returns:
            None. Writes the ONNX model to `out`.

        """
        # Set up logging.
        _warnings.filter_export_warnings()
        _logging.set_up_console_logging()

        # Set the model in eval and deploy mode.
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

        # Try to infer num_channels if not provided.
        if num_channels is None:
            if self.image_normalize is not None:
                num_channels = len(self.image_normalize["mean"])
                logger.info(
                    f"Inferred num_channels={num_channels} from image_normalize."
                )
            else:
                # Try to find the number of channels from the first convolutional layer.
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

        if dynamic_batch_size:
            batch_size = 2
        dynamic_axes = {"images": {0: "N"}} if dynamic_batch_size else None

        # Create dummy input using same device and dtype as the model.
        dummy_input = torch.randn(
            batch_size,
            num_channels,
            self.image_size[
                0
            ],  # TODO(Thomas, 12/25): Allow passing different image size.
            self.image_size[1],
            requires_grad=False,
            device=model_device,
            dtype=torch.float32,
        )

        # TODO(Thomas, 12/25): Add warm-up forward if needed.

        # Set the input/output names.
        input_names = ["images"]
        output_names = self.get_export_output_names()

        # TODO(Nauryzbay, 05/2026): When refactoring forward() to use forward_backend(),
        # expose orig_target_size as a second ONNX input to rescale boxes to original
        # image coordinates inside the graph.
        torch.onnx.export(
            self,
            (dummy_input,),
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
            import onnxslim  # type: ignore [import-not-found,import-untyped]

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
                reference_outputs = reference_model(
                    dummy_input.cpu().to(torch.float32),
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

                    # Due to the presence of top-k operations in the model, the outputs may be
                    # in different order but still valid. To account for this, we sum
                    # over the query dimension before comparing.
                    output_model = output_model.sum(dim=1)
                    if output_onnx.is_floating_point:
                        # Convert to fp32 to avoid overflow issues when summing in fp16.
                        output_onnx = output_onnx.float()
                    output_onnx = output_onnx.sum(dim=1)

                    if output_model.is_floating_point:
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
                        _torch_testing.assert_most_equal(
                            output_onnx,
                            output_model,
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
            # We convert the fp32 attention scores already during ONNX export, so we
            # build a strongly-typed engine: TensorRT then honors those fp32 Cast nodes
            # instead of forcing the whole attention into FP16 (which overflows to NaN).
            fp32_attention_scores=False,
            strongly_typed=True,
            verbose=verbose,
        )
