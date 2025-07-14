#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)

MODEL_NAME = "vitb14"

export_path = f"dinov2_vit_ckpts/{MODEL_NAME}/exported_models"
backbone_weights_path = f"{export_path}/exported_last.pt"
onnx_path = f"{export_path}/dinov2_segmentation_{MODEL_NAME}.onnx"

torch.use_deterministic_algorithms(True)

# Load the model
model = DINOv2SemanticSegmentation(
    model_name=MODEL_NAME,
    num_classes=2,
    model_args={
        "backbone_weights": backbone_weights_path,
    },
)
model.eval()

# Load and sanity-check the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_path, full_check=True)

# Compare the ONNX model output with the PyTorch model output
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
with torch.no_grad():
    torch_output = model(dummy_input)
torch_output_np = torch_output.cpu().numpy()

ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_inputs = {"input": dummy_input.cpu().numpy()}
ort_outs = ort_session.run(["output"], ort_inputs)
onnx_output = ort_outs[0]

np.testing.assert_allclose(
    torch_output_np,
    onnx_output,
    atol=5e-06,  # absolute tolerance
)
