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
    def forward(
        self, x: Tensor, orig_target_size: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class TorchBackend(nn.Module):
    """PyTorch inference backend owning backbone, encoder, decoder, and postprocessor."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        postprocessor: nn.Module,
        internal_class_to_class: Tensor,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.postprocessor = postprocessor
        self.register_buffer(
            "internal_class_to_class",
            internal_class_to_class,
            persistent=False,
        )

    @property
    def deploy_mode(self) -> bool:
        return self.postprocessor.deploy_mode  # type: ignore[no-any-return]

    def deploy(self) -> None:
        self.postprocessor.deploy()  # type: ignore[no-untyped-call]
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(
        self,
        x: Tensor,
        orig_target_size: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if orig_target_size is None:
            h, w = x.shape[-2:]
            orig_target_size_ = torch.tensor([[w, h]]).to(x.device)
        else:
            # Input is (H, W); postprocessor expects (W, H).
            orig_target_size = orig_target_size[:, [1, 0]]
            orig_target_size_ = orig_target_size.to(device=x.device, dtype=torch.int64)

        feats = self.backbone(x)
        feats = self.encoder(feats)
        feats = self.decoder(feats)

        result: tuple[Tensor, Tensor, Tensor] = self.postprocessor(feats, orig_target_size_)
        assert isinstance(result, tuple)
        labels, boxes, scores = result
        labels = self.internal_class_to_class[labels]  # type: ignore[index]
        return (labels, boxes, scores)

    def _forward_raw(self, x: Tensor) -> Any:
        """Return raw decoder output dict (training / postprocess path)."""
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)

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

    def forward(
        self,
        x: Tensor,
        orig_target_size: Tensor | None = None,  # ignored — baked into the ONNX graph
    ) -> tuple[Tensor, Tensor, Tensor]:
        del orig_target_size
        x_np = x.cpu().numpy()
        outputs = self._session.run(None, {"images": x_np})
        labels = torch.from_numpy(outputs[0]).to(self._device)
        boxes = torch.from_numpy(outputs[1]).to(self._device)
        scores = torch.from_numpy(outputs[2]).to(self._device)
        return (labels, boxes, scores)


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

    def forward(
        self,
        x: Tensor,
        orig_target_size: Tensor | None = None,  # ignored — baked into the TRT engine
    ) -> tuple[Tensor, Tensor, Tensor]:
        del orig_target_size
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
            buf = torch.empty(shape, dtype=self._output_dtypes[name], device=self._device)
            outputs[name] = buf
            self._context.set_tensor_address(name, buf.data_ptr())

        stream = torch.cuda.current_stream(self._device).cuda_stream
        self._context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize(self._device)

        return (outputs["labels"], outputs["boxes"], outputs["scores"])


def _trt_dtype_to_torch(trt: Any, trt_dtype: Any) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    return mapping.get(trt_dtype, torch.float32)
