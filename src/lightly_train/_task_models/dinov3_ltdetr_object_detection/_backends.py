#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class InferenceBackend(Protocol):
    def forward(self, x: Tensor) -> dict[str, Tensor]: ...


class TorchBackend(nn.Module):
    """PyTorch inference backend owning backbone, encoder, and decoder."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self._deployed: bool = False

    def deploy(self) -> None:
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        self._deployed = True

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)  # type: ignore[return-value]

    def forward_train(self, x: Tensor, targets: Any) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x


class ONNXBackend:
    """ONNX Runtime inference backend. Not an nn.Module — holds no torch parameters."""

    def __init__(
        self,
        session_path: PathLike,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found,import-untyped]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "onnxruntime is required for the ONNX backend but is not installed. "
                "Install it via: pip install onnxruntime"
            ) from e
        self._session = ort.InferenceSession(str(session_path))
        self._device = device
        self._dtype = dtype

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x_np = x.cpu().numpy()
        pred_logits_np, pred_boxes_np = self._session.run(
            ["pred_logits", "pred_boxes"], {"images": x_np}
        )
        return {
            "pred_logits": torch.from_numpy(pred_logits_np).to(self._device),
            "pred_boxes": torch.from_numpy(pred_boxes_np).to(self._device),
        }


class TensorRTBackend:
    """TensorRT inference backend. Not an nn.Module — holds no torch parameters."""

    def __init__(
        self,
        engine_path: PathLike,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        try:
            import tensorrt as trt  # type: ignore[import-untyped,import-not-found]
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "TensorRT is required for the TensorRT backend but is not installed.\n"
                "Install TensorRT for your system by following NVIDIA's guide:\n"
                "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html"
            ) from e

        self._device = device
        self._dtype = dtype
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self._trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        self._input_name = "images"
        self._output_names: list[str] = []
        self._output_dtypes: dict[str, torch.dtype] = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)
                self._output_dtypes[name] = _trt_dtype_to_torch(
                    trt, self._engine.get_tensor_dtype(name)
                )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = x.to(device=self._device, dtype=self._dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        batch_size = x.shape[0]
        self._context.set_input_shape(self._input_name, tuple(x.shape))
        self._context.set_tensor_address(self._input_name, x.data_ptr())

        outputs: dict[str, Tensor] = {}
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            # Replace any still-dynamic dim with the actual batch size.
            shape = tuple(batch_size if s == -1 else s for s in shape)
            buf = torch.empty(
                shape, dtype=self._output_dtypes[name], device=self._device
            )
            outputs[name] = buf
            self._context.set_tensor_address(name, buf.data_ptr())

        stream = torch.cuda.current_stream(self._device).cuda_stream
        self._context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize(self._device)

        return {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }


def _trt_dtype_to_torch(trt: Any, trt_dtype: Any) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    return mapping.get(trt_dtype, torch.float32)
