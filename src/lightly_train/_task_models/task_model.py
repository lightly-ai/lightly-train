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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Literal,
)

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.export import Dim
from torch.export.dynamic_shapes import Dim as ExportDim
from torch.export.dynamic_shapes import _DimHint
from torch.nn import Module
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._events import tracker
from lightly_train._export.onnx_helpers import (
    check_onnx_dynamo_requirements,
    fix_topological_order,
    remove_redundant_casts,
)
from lightly_train._task_models import task_model_io
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    ModelInputSpec,
    TensorSpec,
)
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

    @property
    def is_deploy_mode(self) -> bool:
        """Whether deploy() has been applied.

        Deploy is an irreversible, in-place transform, so this only ever goes
        False -> True. Defaults to False; task models that support deploy override it.
        """
        return False

    def deploy(self) -> Self:
        """Optimize the model in place for inference (irreversible).

        No-op by default. Task models that reparameterize for deployment override this.
        """
        return self

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


@dataclass(frozen=True)
class ONNXExportPrecisionPolicy:
    """Precision policy for ONNX export.

    Args:
        fp32_module_names:
            Fully-qualified module names or module name prefixes to keep in FP32 after
            casting the model to the requested export precision.
        fp32_module_types:
            Module classes to keep in FP32 after casting the model to the requested
            export precision.
        fp32_onnx_op_types:
            ONNX operator types to keep in FP32 when converting an exported graph to
            FP16.
    """

    fp32_module_names: tuple[str, ...] = ()
    fp32_module_types: tuple[type[Module], ...] = ()
    fp32_onnx_op_types: tuple[str, ...] = ()


class ExportMixin(ABC):
    @property
    @abstractmethod
    def model_input_spec(self) -> ModelInputSpec: ...

    @property
    def onnx_export_precision_policy(self) -> ONNXExportPrecisionPolicy:
        return ONNXExportPrecisionPolicy()

    @abstractmethod
    def verify_onnx_export_outputs(
        self,
        *,
        torch_outputs: BaseModelOutput,
        onnx_outputs: BaseModelOutput,
    ) -> None: ...

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
        # Validate the passed arguments via the config (strict, forbids extras).
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
        check_onnx_dynamo_requirements()

        if not isinstance(self, TaskModel):
            raise TypeError("ExportMixin can only be used with TaskModel subclasses.")

        module: TaskModel = self

        first_parameter = next(module.parameters(), None)
        device = (
            first_parameter.device
            if first_parameter is not None
            else torch.device("cpu")
        )

        if config.precision == "fp32":
            dtype: torch.dtype | None = torch.float32
        elif config.precision == "fp16":
            dtype = torch.float16
        elif config.precision == "auto":
            dtype = None
        else:
            raise ValueError(
                f"Invalid precision '{config.precision}'. Must be one of 'auto', 'fp32', 'fp16'."
            )

        module.eval()
        module.deploy()

        self._apply_onnx_export_module_precision(
            module=module,
            dtype=dtype,
        )

        spec = self.model_input_spec
        export_spec = self._onnx_export_input_spec(
            spec=spec,
            height=config.height,
            width=config.width,
        )
        default_batch_size = 2 if config.dynamic_batch_size else 1
        batch = (
            config.batch_size if config.batch_size is not None else default_batch_size
        )

        example_inputs = self._onnx_export_example_inputs(
            spec=export_spec,
            batch_size=batch,
            dtype=dtype,
            device=device,
        )

        # Build the dynamic shapes from the spec, forcing a static batch dim if
        # dynamic batching is disabled.
        dynamic_shapes: dict[str, tuple[_DimHint | ExportDim, ...]] = {}
        for name, dims in export_spec.input_dynamic_shapes.items():
            new_dims = list(dims)
            if (
                not config.dynamic_batch_size
                and export_spec.input_specs[name].is_batched
            ):
                new_dims[0] = Dim.STATIC
            dynamic_shapes[name] = tuple(new_dims)

        input_names = list(export_spec.input_specs)

        with torch.no_grad():
            example_output = module(**example_inputs)
        output_names = [field.name for field in fields(example_output)]

        torch.onnx.export(
            module,
            args=(),
            kwargs=example_inputs,
            f=str(config.out),
            input_names=input_names,
            output_names=output_names,
            opset_version=config.opset_version,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
            **(config.format_args or {}),
        )

        self._apply_onnx_export_graph_precision(
            out=config.out,
            precision=config.precision,
        )

        if config.simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                model=str(config.out),
                output_model=str(config.out),
                # We skip constant folding as this currently increases the model size by
                # quite a lot.
                skip_optimizations=["constant_folding"],
            )

        if config.verify:
            self._verify_onnx_export(
                out=config.out,
                module=module,
                example_inputs=example_inputs,
            )

        logger.info(f"Successfully exported ONNX model to '{config.out}'")

    def _onnx_export_input_spec(
        self,
        *,
        spec: ModelInputSpec,
        height: int | None,
        width: int | None,
    ) -> ModelInputSpec:
        """Return a concrete export input spec with export-time spatial overrides."""
        input_specs: dict[str, TensorSpec] = {}
        for name, tensor_spec in spec.input_specs.items():
            shape = list(tensor_spec.shape)
            if len(shape) >= 2:
                if height is not None:
                    shape[-2] = height
                if width is not None:
                    shape[-1] = width
            input_specs[name] = tensor_spec.model_copy(update={"shape": tuple(shape)})
        return spec.model_copy(update={"input_specs": input_specs})

    def _onnx_export_example_inputs(
        self,
        *,
        spec: ModelInputSpec,
        batch_size: int,
        dtype: torch.dtype | None,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """Build tracing inputs from a concrete export input spec."""
        example_inputs: dict[str, Tensor] = {}
        for name, tensor in spec.example_inputs(batch_size=batch_size).items():
            if tensor.is_floating_point():
                tensor_dtype = dtype if dtype is not None else tensor.dtype
                tensor = torch.randn(
                    tensor.shape,
                    dtype=tensor_dtype,
                    device=device,
                )
            else:
                tensor = tensor.to(device=device)
            example_inputs[name] = tensor
        return example_inputs

    def _apply_onnx_export_module_precision(
        self,
        *,
        module: Module,
        dtype: torch.dtype | None,
    ) -> None:
        """Apply the requested module precision and FP32 module overrides."""
        if dtype is not None:
            module.to(dtype)
        if dtype != torch.float16:
            return
        policy = self.onnx_export_precision_policy
        for name, child_module in module.named_modules():
            if self._is_onnx_export_fp32_module_name(
                name=name,
                fp32_module_names=policy.fp32_module_names,
            ) or isinstance(child_module, policy.fp32_module_types):
                child_module.to(torch.float32)

    @staticmethod
    def _is_onnx_export_fp32_module_name(
        *,
        name: str,
        fp32_module_names: tuple[str, ...],
    ) -> bool:
        """Return whether ``name`` is a configured FP32 module or descendant."""
        return any(
            name == fp32_name or name.startswith(f"{fp32_name}.")
            for fp32_name in fp32_module_names
        )

    def _apply_onnx_export_graph_precision(
        self,
        *,
        out: str | Path,
        precision: Literal["auto", "fp32", "fp16"],
    ) -> None:
        """Apply ONNX graph precision policy after export."""
        policy = self.onnx_export_precision_policy
        if precision != "fp16" or not policy.fp32_onnx_op_types:
            return

        import onnx
        from onnxruntime.transformers import float16 as ort_float16

        model_onnx = onnx.load(str(out))
        op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + list(
            policy.fp32_onnx_op_types
        )
        model_fp16 = ort_float16.convert_float_to_float16(
            model_onnx,
            op_block_list=op_block_list,
        )
        remove_redundant_casts(model_fp16)
        fix_topological_order(model_fp16)
        onnx.save(model_fp16, str(out))

    def _verify_onnx_export(
        self,
        *,
        out: str | Path,
        module: Module,
        example_inputs: dict[str, Tensor],
    ) -> None:
        """Validate the ONNX file and compare ONNX Runtime outputs to PyTorch."""
        logger.info("Verifying ONNX model")
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(str(out), full_check=True)

        # Always run the reference input in float32 and on cpu for consistency.
        reference_model = copy.deepcopy(module).cpu().to(torch.float32).eval()
        if isinstance(reference_model, TaskModel):
            reference_model.deploy()
        reference_inputs = {
            name: (
                tensor.detach().cpu().to(torch.float32)
                if tensor.is_floating_point()
                else tensor.detach().cpu()
            )
            for name, tensor in example_inputs.items()
        }
        reference_output: BaseModelOutput = reference_model(**reference_inputs)
        reference_values, context = task_model_io._model_output_flatten(
            reference_output
        )

        # Get outputs from the ONNX model.
        session = ort.InferenceSession(str(out))
        get_session_inputs = getattr(session, "get_inputs", lambda: [])
        session_input_types = {
            input_.name: input_.type for input_ in get_session_inputs()
        }
        input_feed = {}
        for name, tensor in example_inputs.items():
            tensor = tensor.detach().cpu()
            if tensor.is_floating_point() and session_input_types.get(name) == (
                "tensor(float16)"
            ):
                tensor = tensor.half()
            input_feed[name] = tensor.numpy()
        outputs_onnx = session.run(output_names=None, input_feed=input_feed)
        onnx_values = [torch.from_numpy(y) for y in outputs_onnx]

        if len(onnx_values) != len(reference_values):
            raise AssertionError(
                f"Number of onnx outputs should be {len(reference_values)} but is "
                f"{len(onnx_values)}"
            )
        onnx_output = task_model_io._model_output_unflatten(
            onnx_values,
            context,
            output_type=type(reference_output),
        )
        self.verify_onnx_export_outputs(
            torch_outputs=reference_output,
            onnx_outputs=onnx_output,
        )
