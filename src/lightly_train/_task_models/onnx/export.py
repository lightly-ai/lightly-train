import glob
import os

import numpy as np
import torch
import onnx
import onnxruntime as ort
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import DINOv2SemanticSegmentation

out_path = "dinov2_vit_ckpts/_vit_test14/exported_models"

model = DINOv2SemanticSegmentation(
    model_name="_vit_test14",
    num_classes=2,
    model_args={
        "backbone_weights": f"{out_path}/exported_last.pt",
    },
)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
onnx_path = f"{out_path}/dinov2_segmentation_test.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['output'],
)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

with torch.no_grad():
    torch_output = model(dummy_input)
ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
ort_inputs = {'input': dummy_input.cpu().numpy()}
ort_outs = ort_session.run(['output'], ort_inputs)
onnx_output = ort_outs[0]

# Convert PyTorch output to NumPy
torch_output_np = torch_output.cpu().numpy()

# Check numerical closeness
np.testing.assert_allclose(
    torch_output_np,
    onnx_output,
    atol=1e-05       # absolute tolerance
)