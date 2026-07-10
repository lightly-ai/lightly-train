#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    cast,
)

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.export import Dim
from torch.export.dynamic_shapes import _DimHint
from torch.nn import Module

from lightly_train._configs.config import PydanticConfig
from lightly_train._events import tracker
from lightly_train._export.onnx_helpers import check_onnx_dynamo_requirements
from lightly_train._task_models import task_model_io
from lightly_train._task_models.task_model_io import BaseModelOutput, ModelInputSpec
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_INFERENCE_TYPE_PATTERNS: dict[str, str] = {
    "ObjectDetection": "object_detection",
    "DepthEstimation": "depth_estimation",
    "SemanticSegmentation": "semantic_segmentation",
    "InstanceSegmentation": "instance_segmentation",
    "PanopticSegmentation": "panoptic_segmentation",
}


class TaskModel(Module):
    """Base class for task-specific models that the user interacts with.

    Must implement the forward method for inference. Must be pure PyTorch and not rely
    on Fabric or Lightning modules.
    """

    model_suffix: str

    def __init__(
        self,
        init_args: dict[str, Any],
        ignore_args: set[str] | None = None,
    ) -> None:
        """
        Args:
            init_args:
                Arguments used to initialize the model. We save those to make it easy
                to serialize and load the model again.
            ignore_args:
                Arguments in init_args that should be ignored. This is useful to ignore
                arguments that are not relevant for serialization, such as
                `backbone_weights` which is not relevant anymore after the model is
                loaded for the first time.
        """
        super().__init__()
        ignore_args = set() if ignore_args is None else ignore_args
        ignore_args.update({"self", "__class__"})
        unknown_keys = ignore_args - init_args.keys()
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in ignore_args: {unknown_keys}. "
                "Please contact the Lightly team if you encounter this error."
            )
        self._init_args = {k: v for k, v in init_args.items() if k not in ignore_args}

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        raise NotImplementedError()

    @property
    def init_args(self) -> dict[str, Any]:
        """Returns the arguments used to initialize the model.

        This is useful for serialization of the model.
        """
        return self._init_args

    @property
    def class_path(self) -> str:
        """Returns the class path of the model.

        This is useful for serialization of the model.
        """
        return f"{self.__module__}.{self.__class__.__name__}"

    def predict(self, image: PathLike | PILImage | Tensor) -> Any:
        """Returns predictions for the given image.

        Args:
            image:
                The input image as a path, URL, PIL image, or tensor. Tensors must have
                shape (C, H, W).
        """
        raise NotImplementedError()

    def preprocess_image(
        self, image: PathLike | PILImage | Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        """Per-image preprocessing producing a tensor and metadata.

        Runs once per input. Output tensors across the batch must share the same
        shape so they can be stacked. Kept separate from `preprocess_batch` so
        non-batchable work stays on the host.
        """
        raise NotImplementedError()

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        """Batch-level preprocessing on a stacked (B, C, H, W) tensor.

        Kept separate from `preprocess_image` so this stage could be baked into an
        exported graph (ONNX/TRT) — torchvision.transforms.v2 ops are batch-
        friendly and run on GPU.
        """
        raise NotImplementedError()

    def forward_backend(self, x: Tensor) -> Any:
        """Run the model on a batched input and return raw outputs (pre-postprocess)."""
        raise NotImplementedError()

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> list[Any]:
        """Map raw outputs and per-image metadata into one result dict per image."""
        raise NotImplementedError()

    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
    ) -> list[Any]:
        """Returns predictions for the given batch of images.

        Args:
            images:
                Sequence of input images. Each can be a path, URL, PIL image, or
                tensor of shape (C, H, W).
        """
        raise NotImplementedError()

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict from a training checkpoint."""
        raise NotImplementedError()

    def _track_inference(self) -> None:
        """Track inference event for analytics.

        This method is called at the start of predict() in subclasses. It is wrapped
        in a try/except to ensure tracking errors never affect the user experience.
        """
        try:
            # Derive task type from class name.
            class_name = self.__class__.__name__
            inference_type = "unknown"
            for pattern, itype in _INFERENCE_TYPE_PATTERNS.items():
                if pattern in class_name:
                    inference_type = itype
                    break

            tracker.track_inference_started(
                task_type=inference_type,
                model=self,
            )
        except Exception:
            # Never let tracking errors affect the user experience.
            pass


class DynamoCompileAble(ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        original = cls.__dict__.get("forward")
        if original is None:
            return

        @functools.wraps(original)
        def checked_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
            return original(self, *args, **kwargs)

        cls.forward = torch.compile(checked_forward, fullgraph=True)  # type: ignore[method-assign]

    @abstractmethod
    def forward(self, *args: Tensor) -> BaseModelOutput:
        """Interface for processors that can be exported to ONNX.

        The processor should take one or more tensors as input and return a subclass of
        ``BaseModelOutput`` as output.
        """
        ...


class DynamoExportConfig(PydanticConfig):
    out: str | Path
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    batch_size: int | None = None
    dynamic_batch_size: bool = True
    height: int | None = None
    width: int | None = None
    opset_version: int | None = None
    simplify: bool = True
    verify: bool = True
    format_args: dict[str, Any] | None = None


class ExportMixin(DynamoCompileAble, ABC):
    @property
    @abstractmethod
    def model_input_spec(self) -> ModelInputSpec: ...

    def export_onnx(
        self,
        out: str | Path,
        *,
        precision: Literal["auto", "fp32", "fp16"] = "auto",
        batch_size: int | None = None,
        dynamic_batch_size: bool = True,
        height: int | None = None,
        width: int | None = None,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
    ) -> None:
        """Exports the processor to ONNX using the dynamo exporter.

        The export is driven by ``self.model_input_spec``: example inputs, input names
        and dynamic shapes are all derived from the spec, and the output names are
        derived from the ``BaseModelOutput`` returned by ``forward``. Every argument
        defaults to "auto" (``None`` / spec-derived) and can be overridden.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. "auto" keeps the current precision,
                "fp32"/"fp16" cast the model and floating-point inputs accordingly.
            batch_size:
                Batch size used for tracing. If None, defaults to 2 when
                ``dynamic_batch_size`` is True, otherwise 1.
            dynamic_batch_size:
                If True, the batch dimension stays dynamic (as declared in the spec).
                If False, the batch dimension is fixed to ``batch_size``.
            height:
                Overrides the height (second-to-last dim) of every spatial input. If
                None, the spec shape is used.
            width:
                Overrides the width (last dim) of every spatial input. If None, the
                spec shape is used.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to ``torch.onnx.export``.

        Returns:
            None. Writes the ONNX model to ``out``.

        Raises:
            pydantic.ValidationError: If any of the passed arguments are invalid.
        """
        # Validate the passed arguments via the config (strict, forbids extras) and
        # read the validated values back from it.
        config = DynamoExportConfig(
            out=out,
            precision=precision,
            batch_size=batch_size,
            dynamic_batch_size=dynamic_batch_size,
            height=height,
            width=width,
            opset_version=opset_version,
            simplify=simplify,
            verify=verify,
            format_args=format_args,
        )
        out = config.out
        precision = config.precision
        batch_size = config.batch_size
        dynamic_batch_size = config.dynamic_batch_size
        height = config.height
        width = config.width
        opset_version = config.opset_version
        simplify = config.simplify
        verify = config.verify
        format_args = config.format_args

        check_onnx_dynamo_requirements()

        # This mixin is always combined with a torch.nn.Module subclass.
        module = cast(Module, self)
        module.eval()

        first_parameter = next(module.parameters(), None)
        device = (
            first_parameter.device
            if first_parameter is not None
            else torch.device("cpu")
        )

        if precision == "fp32":
            dtype: torch.dtype | None = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision == "auto":
            dtype = None
        else:
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'auto', 'fp32', 'fp16'."
            )

        if dtype is not None:
            module.to(dtype)

        spec = self.model_input_spec
        default_batch_size = 2 if dynamic_batch_size else 1
        batch = batch_size if batch_size is not None else default_batch_size

        # Build the example inputs from the spec, applying overrides where requested.
        example_inputs: dict[str, Tensor] = {}
        for name, tensor_spec in spec.input_specs.items():
            shape = list(tensor_spec.shape)
            if len(shape) >= 2:
                if height is not None:
                    shape[-2] = height
                if width is not None:
                    shape[-1] = width
            if tensor_spec.is_batched:
                shape = [batch, *shape]

            tensor_dtype = tensor_spec.dtype
            if tensor_dtype.is_floating_point:
                if dtype is not None:
                    tensor_dtype = dtype
                example_inputs[name] = torch.randn(
                    shape, dtype=tensor_dtype, device=device
                )
            else:
                example_inputs[name] = torch.zeros(
                    shape, dtype=tensor_dtype, device=device
                )

        # Build the dynamic shapes from the spec, forcing a static batch dim if
        # dynamic batching is disabled.
        dynamic_shapes: dict[str, tuple[_DimHint, ...]] = {}
        for name, dims in spec.input_dynamic_shapes.items():
            new_dims = list(dims)
            if not dynamic_batch_size and spec.input_specs[name].is_batched:
                new_dims[0] = Dim.STATIC
            dynamic_shapes[name] = tuple(new_dims)

        input_names = list(spec.input_specs)

        # Derive the output names from the BaseModelOutput returned by forward.
        with torch.no_grad():
            example_output = module(**example_inputs)
        output_names = task_model_io.output_names_from_model_output(example_output)

        torch.onnx.export(
            module,
            args=(),
            kwargs=example_inputs,
            f=str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            **(format_args or {}),
        )

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                model=str(out),
                output_model=str(out),
                # We skip constant folding as this currently increases the model size by
                # quite a lot.
                skip_optimizations=["constant_folding"],
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            from lightly_train import _torch_testing

            onnx.checker.check_model(str(out), full_check=True)

            # Always run the reference input in float32 and on cpu for consistency.
            reference_model = copy.deepcopy(module).cpu().to(torch.float32).eval()
            reference_inputs = {
                name: (
                    tensor.detach().cpu().to(torch.float32)
                    if tensor.is_floating_point()
                    else tensor.detach().cpu()
                )
                for name, tensor in example_inputs.items()
            }
            reference_output = reference_model(**reference_inputs)
            reference_values, _ = task_model_io._model_output_flatten(reference_output)

            # Get outputs from the ONNX model.
            session = ort.InferenceSession(str(out))
            input_feed = {
                name: tensor.detach().cpu().numpy()
                for name, tensor in example_inputs.items()
            }
            outputs_onnx = session.run(output_names=None, input_feed=input_feed)
            outputs_onnx = tuple(torch.from_numpy(y) for y in outputs_onnx)

            if len(outputs_onnx) != len(reference_values):
                raise AssertionError(
                    f"Number of onnx outputs should be {len(reference_values)} but is "
                    f"{len(outputs_onnx)}"
                )
            for output_onnx, output_model, output_name in zip(
                outputs_onnx, reference_values, output_names
            ):

                def msg(s: str) -> str:
                    return f'ONNX validation failed for output "{output_name}": {s}'

                if output_model.is_floating_point():
                    # Absolute and relative tolerances are a bit arbitrary and taken from:
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
