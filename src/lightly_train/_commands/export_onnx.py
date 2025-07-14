#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
torch.onnx.export(
    model,
    (dummy_input,),
    onnx_path,
    input_names=["input"],
    output_names=["output"],
)
