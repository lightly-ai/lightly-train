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
import os
from typing import TYPE_CHECKING, Any, Literal

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Linear, Module
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train import _logging, _torch_helpers, _torch_testing
from lightly_train._data import file_helpers
from lightly_train._export import onnx_helpers, tensorrt_helpers
from lightly_train._models import package_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

if TYPE_CHECKING:
    import tensorrt as trt  # type: ignore[import-untyped,import-not-found]

logger = logging.getLogger(__name__)


class ImageClassification(TaskModel):
    model_suffix = "classification"

    def __init__(
        self,
        *,
        # TODO(Guarin, 02/26): Support passing WrappedModel directly.
        model: str,
        classes: dict[int, str],
        classification_task: Literal["multiclass", "multilabel"],
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None,
        backbone_freeze: bool,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        """
        Args:
            model:
                A string specifying the model name. It must be in the
                format "{package_name}/{backbone_name}". For example, "dinov3/vitt16" or
                "dinov3/vitt16-classification". The -classification suffix is optional
                as it is implied by this class.
            classification_task:
                The type of classification task. Either "multiclass" or "multilabel".
                "multiclass" means that each image belongs to exactly one class.
                "multilabel" means that each image can belong to multiple classes.
            classes:
                A dict mapping the class ID to the class name.
            image_size:
                The size of the input images.
            image_normalize:
                A dict containing the mean and standard deviation for normalizing
                the input images. The dict must contain the keys "mean" and "std".
                Example: {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}.
                This is used to normalize the input images before passing them to the
                model. If None, no normalization is applied.
            backbone_freeze:
                If True, the backbone weights are frozen and only the classification
                head is trained.
            backbone_weights:
                Optional path to a checkpoint file containing the backbone weights. If
                provided, the weights are loaded into the model passed via `model`.
            backbone_args:
                Additional arguments for the backbone. Only used if `model` is a string.
                The arguments are passed to the model when it is instantiated.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        super().__init__(locals(), ignore_args={"backbone_weights", "load_weights"})
        parsed_name = self.parse_model_name(model=model)
        self.model_name = parsed_name["model_name"]
        self.classification_task = classification_task
        self.classes = classes
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.backbone_freeze = backbone_freeze

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        # An additional class is added to represent "unknown/ignored classes" if needed.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )

        num_input_channels = (
            3 if self.image_normalize is None else len(self.image_normalize["mean"])
        )

        backbone_model_args = {}
        if model.startswith("dinov2/"):
            backbone_model_args["drop_path_rate"] = 0.0
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)

        self.backbone = package_helpers.get_wrapped_model(
            model=parsed_name["backbone_name"],
            num_input_channels=num_input_channels,
            model_args=backbone_model_args,
            load_weights=load_weights and backbone_weights is None,
        )

        try:
            mask_token = self.backbone.mask_token  # type: ignore[attr-defined]
        except AttributeError:
            pass
        else:
            # TODO(Guarin, 02/26): Improve how mask tokens are handled for fine-tuning.
            # The classification model does not use the mask token. We disable grads
            # for the mask token to avoid issues with DDP and find_unused_parameters.
            mask_token.requires_grad = False

        # Load the backbone weights if a path is provided.
        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(backbone_weights)

        if self.backbone_freeze:
            self.freeze_backbone()

        feature_dim = self.backbone.feature_dim()
        self.class_head = Linear(feature_dim, len(self.classes))
        self.class_head.weight.data.normal_(mean=0.0, std=0.01)
        self.class_head.bias.data.zero_()

        _torch_helpers.register_load_state_dict_pre_hook(
            self, class_head_reuse_or_reinit_hook
        )

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}" for name in package_helpers.list_model_names()
        ]

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        # This class works with any model.
        return True

    @classmethod
    def parse_model_name(cls, model: str) -> dict[str, str]:
        model_name = model
        backbone_name = model
        # Suffix is optional as this class supports any backbone model.
        if model.endswith(f"-{cls.model_suffix}"):
            backbone_name = model[: -len(f"-{cls.model_suffix}")]
        else:
            model_name = f"{model}-{cls.model_suffix}"

        return {
            "model_name": model_name,
            "backbone_name": backbone_name,
        }

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, topk: int = 1, threshold: float = 0.5
    ) -> dict[str, Tensor]:
        """Returns the predicted labels and scores for the input image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape (C, H, W).

        Returns:
            A {"labels": Tensor, "scores": Tensor} dict. Labels has shape (topk,) for
            multiclass and (num_labels,) for multilabel where num_labels is the number
            of labels with score > threshold. Scores has the same shape as labels and
            contains the corresponding scores.
        """
        self._track_inference()
        if self.training:
            self.eval()

        first_param = next(self.parameters())
        device = first_param.device
        dtype = first_param.dtype

        # Load image
        x = file_helpers.as_image_tensor(image).to(device)

        # Transform
        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        if self.image_normalize is not None:
            x = transforms_functional.normalize(
                x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
            )
        x = self.resize_and_pad(x)[0]
        x = x.unsqueeze(0)  # (1, C, H', W')

        # Forward
        logits = self.forward_train(x)  # (B, num_classes)
        labels, scores = self.get_labels_scores(logits, topk=topk, threshold=threshold)

        if self.classification_task == "multiclass":
            labels = self.internal_class_to_class[labels]
        elif self.classification_task == "multilabel":
            labels = self.internal_class_to_class[labels[..., 1]]
        else:
            raise ValueError(
                f"Invalid classification_task '{self.classification_task}'"
            )

        # Remove batch dimension.
        if self.classification_task == "multiclass":
            labels = labels.squeeze(0)  # Remove batch dimension
            scores = scores.squeeze(0)  # Remove batch dimension
        elif self.classification_task == "multilabel":
            labels = labels[..., 1]  # Keep label column and remove batch indices

        return {
            "labels": labels,
            "scores": scores,
        }

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward for ONNX export.

        Returns:
            (labels, scores) tuple. Labels has shape (B, topk) for multiclass and
            (num_labels, 2) for multilabel where the columns are (batch_idx, label).
            Scores has shape (B, topk) for multiclass and (num_labels,) for multilabel.
        """
        logits = self.forward_train(x)
        labels, scores = self.get_labels_scores(logits, topk=1, threshold=-1)
        if self.classification_task == "multiclass":
            labels = self.internal_class_to_class[labels]
        elif self.classification_task == "multilabel":
            labels[..., 1] = self.internal_class_to_class[labels[..., 1]]
        else:
            raise ValueError(
                f"Invalid classification_task '{self.classification_task}'"
            )
        return labels, scores

    def forward_train(self, x: Tensor) -> Tensor:
        """Forward pass for training. Returns the class logits."""
        features = self.backbone.forward_pool(self.backbone.forward_features(x))
        x = features["pooled_features"]  # (B, C, H, W)
        x = self.class_head(x.flatten(start_dim=1))  # (B, num_classes)
        return x

    def get_labels_scores(
        self, logits: Tensor, topk: int, threshold: float
    ) -> tuple[Tensor, Tensor]:
        """Returns the predicted labels and scores for the input logits.

        Returns:
            (labels, scores) tuple. Labels has shape (B, topk) for multiclass and
            (num_labels, 2) for multilabel where the columns are (batch_idx, label).
            Scores has shape (B, topk) for multiclass and (num_labels,) for multilabel.
        """
        if self.classification_task == "multiclass":
            scores = logits.softmax(dim=-1)  # (B, num_classes)
            scores, labels = torch.topk(scores, k=topk, dim=-1)  # (B, topk), (B, topk)
        elif self.classification_task == "multilabel":
            scores = logits.sigmoid()  # (B, num_classes)
            keep = scores > threshold  # (B, num_classes)
            # (num_labels, 2) where columns are (batch_idx, label)
            labels = keep.nonzero(as_tuple=False)
            scores = scores[keep]  # (num_labels,)
        else:
            raise ValueError(
                f"Invalid classification_task '{self.classification_task}'"
            )
        return labels, scores

    def resize_and_pad(self, image: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Resize and pad image to self.image_size while keeping aspect ratio constant.

        Args:
            image:
                A tensor of shape (..., H, W).

        Returns:
            An (image, (crop_h, crop_w)) tuple where image is a tensor of shape
            (..., H', W') with H'==self.image_size[0] and W'==self.image_size[1], and
            (crop_h, crop_w) are the height and width of the resized (non-padded) image.
        """
        image_h, image_w = image.shape[-2:]
        resize_factor = min(self.image_size[0] / image_h, self.image_size[1] / image_w)
        crop_h = round(image_h * resize_factor)
        crop_w = round(image_w * resize_factor)
        pad_h = max(0, self.image_size[0] - crop_h)
        pad_w = max(0, self.image_size[1] - crop_w)
        # (..., crop_h, crop_w)
        image = transforms_functional.resize(image, size=[crop_h, crop_w])
        # (..., H', W')
        image = transforms_functional.pad(image, padding=[0, 0, pad_w, pad_h])
        return image, (crop_h, crop_w)

    def load_backbone_weights(self, path: PathLike) -> None:
        """
        Load backbone weights from a checkpoint file.

        Args:
            path: path to a .pt file, e.g., exported_last.pt.
        """
        # Check if the file exists.
        if not os.path.exists(path):
            logger.error(f"Checkpoint file not found: {path}")
            return

        # Load the checkpoint.
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        unwrapped_backbone = self.backbone.get_model()
        missing, unexpected = unwrapped_backbone.load_state_dict(
            state_dict, strict=False
        )

        # Log missing and unexpected keys.
        if missing or unexpected:
            if missing:
                logger.warning(f"Missing keys when loading backbone: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading backbone: {unexpected}")
        else:
            logger.info(f"Backbone weights loaded from '{path}'")

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("model."):
                name = name[len("model.") :]
                new_state_dict[name] = param
        self.load_state_dict(new_state_dict, strict=True)

    @torch.no_grad()
    def export_onnx(
        self,
        out: PathLike,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        batch_size: int = 1,
        height: int | None = None,
        width: int | None = None,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> None:
        """Exports the model to ONNX for inference.

        The export uses a dummy input of shape (batch_size, C, H, W) where C is inferred
        from the first model parameter and (H, W) come from `self.image_size`.
        The ONNX graph uses dynamic batch size for both inputs and produces
        two outputs: labels and scores.

        Optionally simplifies the exported model in-place using onnxslim and
        verifies numerical closeness against a float32 CPU reference via
        ONNX Runtime.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. Either "auto", "fp32", or "fp16". "auto"
                uses the model's current precision.
            batch_size:
                Batch size for the ONNX input.
            height:
                Height of the ONNX input. If None, will be taken from `self.image_size`.
            width:
                Width of the ONNX input. If None, will be taken from `self.image_size`.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to `torch.onnx.export`.

        Returns:
            None. Writes the ONNX model to `out`.
        """
        # TODO(Guarin, 01/26): Move warnings module out of commands subpackage and
        # move import to the top of the file.
        from lightly_train._commands import _warnings

        _logging.set_up_console_logging()
        _warnings.filter_export_warnings()

        self.eval()

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

        height = self.image_size[0] if height is None else height
        width = self.image_size[1] if width is None else width
        num_channels = (
            3 if self.image_normalize is None else len(self.image_normalize["mean"])
        )

        dummy_input = torch.randn(
            batch_size,
            num_channels,
            height,
            width,
            requires_grad=False,
            device=model_device,
            dtype=dtype,
        )

        # Precalculate interpolated positional encoding for ONNX export.
        with onnx_helpers.precalculate_for_onnx_export():
            self(dummy_input)

        input_names = ["images"]
        output_names = ["labels", "scores"]

        torch.onnx.export(
            self,
            (dummy_input,),
            str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes={"images": {0: "N"}},
            **(format_args or {}),
        )

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            # Simplify.
            onnxslim.slim(
                model=str(out),
                output_model=out,
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(out, full_check=True)

            # Always run the reference input in float32 and on cpu for consistency.
            reference_model = copy.deepcopy(self).cpu().to(torch.float32).eval()
            reference_outputs = reference_model(
                dummy_input.cpu().to(torch.float32),
            )

            # Get outputs from the ONNX model.
            session = ort.InferenceSession(out)
            input_feed = {
                "images": dummy_input.cpu().numpy(),
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
            See the `TensorRT documentation <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>`_ for more details.
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

        def update_network_fn(net: trt.INetworkDefinition) -> None:
            import tensorrt as trt

            # TODO(Guarin, 02/26): Check if we have to handle sigmoid/softmax.
            wanted = ("ReduceSum", "Div", "Mul", "Sigmoid", "Softmax")

            io_tensors = {
                *(net.get_input(i) for i in range(net.num_inputs)),
                *(net.get_output(i) for i in range(net.num_outputs)),
            }

            # find first Sigmoid layer index
            start_idx: int | None = None
            for i in range(net.num_layers):
                layer = net.get_layer(i)
                # TODO(Guarin, 02/26): Check if we have to handle sigmoid/softmax.
                if "Sigmoid" in layer.name or "Softmax" in layer.name:
                    start_idx = i
                    break
            if start_idx is None:
                logger.warning("No Sigmoid or Softmax layer found; nothing to update.")
                return

            for i in range(start_idx, net.num_layers):
                layer = net.get_layer(i)
                if any(k in layer.name for k in wanted):
                    layer.precision = trt.DataType.FLOAT

                    for j in range(layer.num_outputs):
                        out = layer.get_output(j)
                        if out is None:
                            continue

                        # Only set dtype for network I/O tensors to avoid TRT warnings.
                        if out in io_tensors:
                            out.dtype = trt.DataType.FLOAT

                    logger.debug(f"Forcing FP32 for layer: {layer.name}")

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
            # FP32 attention scores required for FP16 model stability. Otherwise output
            # contains NaN.
            fp32_attention_scores=True,
            verbose=verbose,
            update_network_fn=update_network_fn,
        )

    def freeze_backbone(self) -> None:
        self.backbone.eval()  # type: ignore[attr-defined]
        for param in self.backbone.parameters():
            param.requires_grad = False


def class_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reuse or reinitialize class head when number of classes changes."""
    class_head_weight_key = f"{prefix}class_head.weight"
    class_head_bias_key = f"{prefix}class_head.bias"
    class_head_weight = state_dict.get(class_head_weight_key)
    if class_head_weight is None:
        return

    class_head_module = getattr(module, "class_head", None)
    if class_head_module is None:
        return

    num_classes_state = class_head_weight.shape[0]
    num_classes_module = class_head_module.out_features
    if num_classes_state == num_classes_module:
        return
    else:
        logger.info(
            f"Checkpoint provides {num_classes_state} classes but module expects "
            f"{num_classes_module}. Reinitializing class head.",
        )

        # Keep the module initialization by overwriting the checkpoint weights with the
        # current parameter tensors.
        state_dict[class_head_weight_key] = class_head_module.weight.detach().clone()
        state_dict[class_head_bias_key] = class_head_module.bias.detach().clone()
