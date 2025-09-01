#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import contextvars
import logging
from abc import abstractmethod
from collections.abc import Iterator, Container, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol, Tuple, runtime_checkable

import torch
from torch import distributed

from lightly_train import _logging
from lightly_train._commands import _warnings, common_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._task_models import task_model_helpers
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_PRECALCULATE_FOR_ONNX_EXPORT = contextvars.ContextVar(
    "PRECALCULATE_FOR_ONNX_EXPORT", default=False
)


def is_in_precalculate_for_onnx_export() -> bool:
    return _PRECALCULATE_FOR_ONNX_EXPORT.get()


@contextlib.contextmanager
def precalculate_for_onnx_export() -> Iterator[None]:
    """
    For certain models we want to precalculate some values and store them in the model before
    exporting the model to ONNX. In order to avoid having to pass that options through all methods we have
    this context manager. Therefore, one should call
    ```
    with precalculate_for_onnx_export():
        model(example_input)
    ```
    before running `torch.onnx.export(model, example_input)`.
    In the relevant part of the model we can check if we are in this context with
    `is_in_precalculate_for_onnx_export()`.
    """
    token = _PRECALCULATE_FOR_ONNX_EXPORT.set(True)
    try:
        yield
    finally:
        _PRECALCULATE_FOR_ONNX_EXPORT.reset(token)


def export_onnx(
    *,
    out: PathLike,
    checkpoint: PathLike,
    batch_size: int = 1,
    num_channels: int = 3,
    height: int = 224,
    width: int = 224,
    half: bool = False,
    verify: bool = True,
    overwrite: bool = False,
    format_args: dict[str, Any] | None = None,
) -> None:
    return _export_task(format="onnx", **locals())


def _export_task(
    *,
    out: PathLike,
    checkpoint: PathLike,
    format: Literal["onnx"],
    batch_size: int = 1,
    num_channels: int = 3,
    height: int = 224,
    width: int = 224,
    half: bool = False,
    verify: bool = True,
    overwrite: bool = False,
    format_args: dict[str, Any] | None = None,
) -> None:
    """Export a model from a checkpoint.

    Args:
        out:
            Path where the exported model will be saved.
        checkpoint:
            Path to the LightlyTrain checkpoint file to export the model from.
        format:
            Format to save the model in.
        batch_size:
            Batch size of the input tensor.
        num_channels:
            Number of channels in input tensor.
        height:
            Height of the input tensor.
        width:
            Width of the input tensor.
        half:
            Export the model with float16 precision.
        verify:
            Check the exported model for errors.
        overwrite:
            Overwrite the output file if it already exists.
        format_args:
            Format specific arguments. Eg. "dynamic" for onnx and int8 precision for tensorrt.
    """
    config = ExportTaskConfig(**locals())
    _export_task_from_config(config=config)


class ONNXPrecision(Enum):
    FP16 = "float16"
    FP32 = "float32"


@runtime_checkable
class ONNXExportable(Protocol):
    """
    A protocol to specify that a model can be exported to ONNX.

    Some default implementations are provided for most methods these can only be used if one inherits from the Protocol.
    Otherwise, one needs to make a call to `Protocol.somemethod(self, ...)` when implementing the protocol.
    """

    def onnx_opset_versions(self) -> Tuple[int, int | None]:
        """
        The range of ONNX opset versions supported by the model.

        Return a tuple where the first element is the lower bound and the second element is the upper bound (inclusive.
        The upper bound can also be None to indicate that there is no upper bound.
        """
        del self
        return (7, None)

    def onnx_precisions(self) -> Container[ONNXPrecision]:
        """
        The precisions that the ONNX model can be exported with.
        """
        del self
        return {ONNXPrecision.FP16, ONNXPrecision.FP32}

    def verify_torch_onnx_export_kwargs(self, **kwargs: dict[str, Any]) -> None:
        """
        Verify additional arguments passed to torch.onnx.export. Should raise an exception if some argument
        is not supported.
        """
        del self
        del kwargs
        return

    def setup_onnx_model(
        self, *, checkpoint: PathLike, precision: ONNXPrecision
    ) -> torch.nn.Module:
        """
        Set up the exact torch model that should be exported with torch.onnx.export.
        """
        del self
        model = task_model_helpers.load_model_from_checkpoint(checkpoint=checkpoint)
        if precision == ONNXPrecision.FP16:
            model = model.half()
        return model

    def setup_validation_model(self, *, checkpoint: PathLike) -> torch.nn.Module:
        """
        Set up the exact torch model that  is used as a reference model to verify the export onnx model.
        """
        del self
        model = task_model_helpers.load_model_from_checkpoint(
            checkpoint=checkpoint, device="cpu"
        )
        return model

    @abstractmethod
    def make_onnx_export_inputs(
        self, *, precision: ONNXPrecision, device: torch.device, **kwargs
    ) -> Mapping[str, torch.Tensor]:
        """
        Create the dummy input tensors that are used during the ONNX export.

        Should return a mapping from input names to tensors.
        """
        ...

    @abstractmethod
    def onnx_output_names(self) -> Sequence[str]:
        """
        Return the names of the ONNX output tensors.
        """
        ...


def _export_task_from_config(config: ExportTaskConfig) -> None:
    # Only export on rank 0.
    if distributed.is_initialized() and distributed.get_rank() > 0:
        return

    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()
    _logging.set_up_filters()
    logger.info(f"Args: {common_helpers.pretty_format_args(args=config.model_dump())}")

    out_path = common_helpers.get_out_path(
        out=config.out, overwrite=config.overwrite
    ).as_posix()  # TODO(Yutong, 07/25): make sure the format corrsponds to the output file extension!
    checkpoint_path = common_helpers.get_checkpoint_path(checkpoint=config.checkpoint)
    task_model = task_model_helpers.load_model_from_checkpoint(
        checkpoint=checkpoint_path
    )

    # Export the model to ONNX format
    # TODO(Yutong, 07/25): support more formats (may use ONNX as the intermediate format)
    if config.format == "onnx":
        if not isinstance(task_model, ONNXExportable):
            raise ValueError(
                f"Model of class {task_model.__class__.__name__} cannot be exported to ONNX."
            )
        opset_version = config.format_args.get("opset_version", 18)
        opset_lower, opset_upper = task_model.onnx_opset_versions()
        # torch.onnx.export requires at least opset version 7
        if opset_version < max(opset_lower, 7):
            raise ValueError(f"Opset must be a at least {opset_lower}.")
        if opset_upper is not None and opset_version > opset_upper:
            raise ValueError(f"Opset can be at most {opset_upper}.")
        precision = ONNXPrecision.FP16 if config.half else ONNXPrecision.FP32
        if precision not in task_model.onnx_precisions():
            raise ValueError(f"Precision {precision.value} is not supported.")

        export_model = task_model.setup_onnx_model(
            checkpoint=checkpoint_path, precision=precision
        )
        export_model_device = next(export_model.parameters()).device

        dummy_inputs = task_model.make_onnx_export_inputs(
            precision=precision,
            device=export_model_device,
            batch_size=config.batch_size,
            num_channels=config.num_channels,
            height=config.height,
            width=config.width,
        )

        input_names = list(dummy_inputs.keys())
        dummy_inputs = tuple(dummy_inputs.values())
        output_names = task_model.onnx_output_names()

        with precalculate_for_onnx_export():
            export_model(*dummy_inputs)
        logger.info(f"Exporting ONNX model to '{out_path}'")
        torch.onnx.export(
            export_model,
            dummy_inputs,
            out_path,
            input_names=input_names,
            output_names=output_names,
            **config.format_args if config.format_args else {},
        )
        del export_model

        if config.verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(out_path, full_check=True)

            onnx_inputs = task_model.make_onnx_export_inputs(
                precision=precision,
                device=torch.device("cpu"),
                batch_size=config.batch_size,
                num_channels=config.num_channels,
                height=config.height,
                width=config.width,
            )
            # Always run the validation input in float32 and on cpu for consistency
            validation_inputs = [v.to(torch.float32) for v in onnx_inputs.values()]
            onnx_inputs = {k: v.numpy() for (k, v) in onnx_inputs.items()}

            validation_model = task_model.setup_validation_model(
                checkpoint=checkpoint_path
            )
            del task_model

            session = ort.InferenceSession(out_path)
            onnx_outputs = session.run(
                output_names=output_names, input_feed=onnx_inputs
            )
            onnx_outputs = tuple(torch.from_numpy(y) for y in onnx_outputs)

            validation_outputs = validation_model(*validation_inputs)

            if len(onnx_outputs) != len(validation_outputs):
                raise AssertionError(
                    f"Number of onnx outputs should be {len(validation_outputs)} but is {len(onnx_outputs)}"
                )
            for output_onnx, output_model, output_name in zip(
                onnx_outputs, validation_outputs, output_names
            ):
                # Absolute and relative tolerances are a bit arbitrary and taken from here:
                #   https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_core.py#L1611-L1618
                torch.testing.assert_close(
                    output_onnx,
                    output_model,
                    msg=lambda s: f'ONNX validation failed for output "{output_name}": {s}',
                    equal_nan=True,
                    check_device=False,
                    check_dtype=False,
                    check_layout=False,
                    atol=5e-3,
                    rtol=1e-1,
                )

        logger.info(f"Successfully exported ONNX model to '{out_path}'")

    else:
        raise ValueError(
            f"Unsupported format: {config.format}. Supported formats: 'onnx'."
        )


class ExportTaskConfig(PydanticConfig):
    out: PathLike
    checkpoint: PathLike
    format: Literal["onnx"]
    batch_size: int = 1
    num_channels: int = 3
    height: int = 224
    width: int = 224
    half: bool = False
    verify: bool = True
    overwrite: bool = False
    format_args: dict[str, Any] | None = (
        None  # TODO(Yutong, 07/25): use Pydantic models for format_args if needed
    )
